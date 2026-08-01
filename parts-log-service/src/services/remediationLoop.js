const fs = require('fs');
const path = require('path');

const config = require('../config');
const { errors } = require('../middleware/errorLogger');
const claudeClient = require('./claudeClient');
const deploy = require('./deploy');
const k8s = require('./k8s');
const postmanRunner = require('./postmanRunner');
const logMonitor = require('./logMonitor');
const { resolveTargetFile, runTests } = require('./aiFixer');

function nowIso() {
  return new Date().toISOString();
}

/**
 * One pass of steps 1-7: retrieve logs, analyze, fix, push to dev, restart,
 * exercise the affected APIs, and watch logs for fallout. Returns an
 * 'outcome' the caller loops on:
 *   - 'unresolved'          no registered file for this part - retrying won't help
 *   - 'fix_failed'          AI/local test gate failed before anything was deployed
 *   - 'revalidation_failed' deployed to dev but live validation found problems
 *   - 'validated'           deployed to dev and live validation passed
 */
async function runOneCycle(errorEntry, deps) {
  // Step 1: retrieve error logs (this error plus recent history for the
  // same part, so the AI has failure context across retries).
  const relatedLogs = errors.find((e) => e.partKey === errorEntry.partKey).slice(-10);

  // Step 2: analyze - resolve which file/component is responsible.
  const targetFile = deps.resolveTargetFile(errorEntry);
  if (!targetFile || !fs.existsSync(targetFile)) {
    return {
      outcome: 'unresolved',
      note: 'No registered, existing source file for this partKey - register one via POST /api/registry.',
    };
  }

  const original = fs.readFileSync(targetFile, 'utf8');

  // Step 3: fix the root cause.
  let fix;
  try {
    fix = await deps.claudeClient.requestFix({
      errorEntry: { ...errorEntry, relatedLogs },
      fileName: path.relative(config.repoRoot, targetFile),
      fileContents: original,
    });
  } catch (err) {
    return { outcome: 'fix_failed', note: `AI request failed: ${err.message}` };
  }

  if (fix.dryRun || !fix.fixedFile) {
    return { outcome: 'unresolved', note: fix.summary };
  }

  // Optional companion test file (new or updated) covering this bug. Guard
  // against a path outside the repo the same way resolveTargetFile does -
  // if it's suspicious, skip the test write rather than fail the whole fix
  // over it.
  let testFileAbs = null;
  let testFileOriginal = null;
  let testFileIsNew = false;
  if (fix.testFilePath && fix.testFile) {
    const resolved = path.resolve(config.repoRoot, fix.testFilePath);
    if (resolved.startsWith(path.resolve(config.repoRoot))) {
      testFileAbs = resolved;
      testFileIsNew = !fs.existsSync(resolved);
      testFileOriginal = testFileIsNew ? null : fs.readFileSync(resolved, 'utf8');
    }
  }

  fs.writeFileSync(targetFile, fix.fixedFile);
  if (testFileAbs) {
    fs.mkdirSync(path.dirname(testFileAbs), { recursive: true });
    fs.writeFileSync(testFileAbs, fix.testFile);
  }

  const unitTest = await deps.runTests();
  if (!unitTest.passed) {
    fs.writeFileSync(targetFile, original); // revert - never push a fix that fails locally
    if (testFileAbs) {
      if (testFileIsNew) fs.rmSync(testFileAbs);
      else fs.writeFileSync(testFileAbs, testFileOriginal);
    }
    return {
      outcome: 'fix_failed',
      note: `Local test suite failed: ${unitTest.output || 'unknown failure'}`,
    };
  }

  // Step 4: push to the development branch.
  const relFile = path.relative(config.repoRoot, targetFile);
  const filesToCommit = testFileAbs
    ? [relFile, path.relative(config.repoRoot, testFileAbs)]
    : [relFile];
  const devDeploy = await deps.deploy.commitAndDeploy({
    files: filesToCommit,
    branch: config.devBranch,
    message: `fix(auto): ${fix.summary || 'AI remediation'} [error ${errorEntry.id}]`,
  });

  // Step 5: restart the affected Kubernetes deployment/pod so the fix is live.
  const restart = await deps.k8s.restartAndWait();
  if (!restart.skipped && restart.error) {
    return {
      outcome: 'revalidation_failed',
      summary: fix.summary,
      devDeploy,
      restart,
      note: `Kubernetes restart failed: ${restart.error}`,
    };
  }

  // Step 6: execute the affected APIs.
  const postman = await deps.postmanRunner.runCollection({
    collectionPath: config.postmanCollectionPath,
    baseUrl: config.postmanBaseUrl,
    filter: errorEntry.partKey,
  });

  // Step 7: monitor application logs during/after the API execution.
  const monitor = await deps.logMonitor.watch({
    partKey: errorEntry.partKey,
    sinceIso: nowIso(),
    windowMs: config.monitorWindowMs,
    pollIntervalMs: config.monitorPollIntervalMs,
    excludeErrorId: errorEntry.id,
  });

  const outcome = postman.allPassed && monitor.clean ? 'validated' : 'revalidation_failed';
  return { outcome, summary: fix.summary, devDeploy, restart, postman, monitor };
}

function describeFailure(cycle) {
  const parts = [];
  if (cycle.note) parts.push(cycle.note);
  const failedApis = ((cycle.postman && cycle.postman.results) || [])
    .filter((r) => !r.ok)
    .map((r) => `${r.method} ${r.url} -> ${r.status ?? r.error}`);
  if (failedApis.length) parts.push(`Failing APIs: ${failedApis.join('; ')}`);
  const newErrors = ((cycle.monitor && cycle.monitor.newErrors) || []).map((e) => e.message);
  if (newErrors.length) parts.push(`New errors observed after deploy: ${newErrors.join('; ')}`);
  return parts.join(' ') || 'Revalidation failed for an unspecified reason.';
}

/**
 * Full 10-step remediation cycle for one captured error:
 * retrieve -> analyze -> fix -> push to dev -> restart -> test APIs ->
 * monitor -> (repeat on failure, feeding the new failure back into the next
 * AI request) -> promote to prod once validated -> keep monitoring for
 * stability.
 */
async function remediate(errorEntry, injected = {}) {
  const deps = {
    resolveTargetFile,
    runTests,
    claudeClient,
    deploy,
    k8s,
    postmanRunner,
    logMonitor,
    ...injected,
  };

  if (!config.aiFixerEnabled) {
    return errors.update(errorEntry.id, { status: 'skipped', note: 'AI_FIXER_ENABLED=false' });
  }

  const history = [];
  let current = errorEntry;

  for (let attempt = 1; attempt <= config.maxRemediationCycles; attempt += 1) {
    errors.update(errorEntry.id, { status: 'fixing', fixAttempts: attempt });

    const cycle = await runOneCycle(current, deps);
    history.push({ attempt, ...cycle });

    if (cycle.outcome === 'unresolved') {
      return errors.update(errorEntry.id, { status: 'unresolved', note: cycle.note, history });
    }

    if (cycle.outcome === 'validated') {
      // Step 9 done (all APIs passed) - promote to prod, no human step.
      const prodDeploy = await deps.deploy.promoteToProd();

      // Step 10: continue monitoring after successful execution to confirm
      // stability before declaring victory.
      const stability = await deps.logMonitor.watch({
        partKey: errorEntry.partKey,
        sinceIso: nowIso(),
        windowMs: config.postDeployMonitorWindowMs,
        pollIntervalMs: config.monitorPollIntervalMs,
        excludeErrorId: errorEntry.id,
      });

      return errors.update(errorEntry.id, {
        status: 'fixed',
        note: cycle.summary,
        history,
        deploy: { dev: cycle.devDeploy, prod: prodDeploy },
        stability,
      });
    }

    // outcome is 'fix_failed' or 'revalidation_failed'
    if (attempt === config.maxRemediationCycles) {
      return errors.update(errorEntry.id, {
        status: 'fix_failed',
        note: `Exceeded MAX_REMEDIATION_CYCLES (${config.maxRemediationCycles}) without all APIs passing. Last failure: ${describeFailure(cycle)}`,
        history,
      });
    }

    // Step 8: loop back with the new failure context for the next attempt.
    current = {
      ...errorEntry,
      message: describeFailure(cycle),
      stack: current.stack,
    };
  }

  // Unreachable (loop always returns), but keeps the function's control
  // flow explicit for readers and linters.
  return errors.getById(errorEntry.id);
}

module.exports = { remediate, runOneCycle, describeFailure };
