const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const config = require('../config');
const { errors } = require('../middleware/errorLogger');
const { registry } = require('../routes/registry');
const claudeClient = require('./claudeClient');
const deploy = require('./deploy');

const execFileAsync = promisify(execFile);

function runTests() {
  const [cmd, ...args] = config.testCommand.split(' ');
  return execFileAsync(cmd, args, { cwd: config.repoRoot }).then(
    () => ({ passed: true }),
    (err) => ({ passed: false, output: `${err.stdout || ''}\n${err.stderr || ''}`.trim() })
  );
}

function resolveTargetFile(errorEntry) {
  if (!errorEntry.partKey) return null;
  const part = registry.findOne(
    (r) => r.key === errorEntry.partKey || (r.aliases || []).includes(errorEntry.partKey)
  );
  if (!part || !part.file) return null;

  const resolved = path.resolve(config.repoRoot, part.file);
  // Guard against a registry entry ever pointing outside the repo.
  if (!resolved.startsWith(path.resolve(config.repoRoot))) return null;
  return resolved;
}

/**
 * Full pipeline for one captured error: locate the file, ask Claude for a
 * fix, apply it, run tests, and only push to prod if tests pass. Every
 * outcome (including dry-run / no-target / test-failure) is recorded back
 * onto the error entry so /api/errors is a complete audit trail.
 */
async function remediate(errorEntry, deps = {}) {
  const client = deps.claudeClient || claudeClient;
  const deployer = deps.deploy || deploy;
  const test = deps.runTests || runTests;

  if (!config.aiFixerEnabled) {
    return errors.update(errorEntry.id, { status: 'skipped', note: 'AI_FIXER_ENABLED=false' });
  }

  errors.update(errorEntry.id, { status: 'fixing', fixAttempts: (errorEntry.fixAttempts || 0) + 1 });

  const targetFile = resolveTargetFile(errorEntry);
  if (!targetFile || !fs.existsSync(targetFile)) {
    return errors.update(errorEntry.id, {
      status: 'unresolved',
      note: 'No registered, existing source file for this partKey - register one via POST /api/registry.',
    });
  }

  const original = fs.readFileSync(targetFile, 'utf8');

  let result;
  try {
    result = await client.requestFix({
      errorEntry,
      fileName: path.relative(config.repoRoot, targetFile),
      fileContents: original,
    });
  } catch (err) {
    return errors.update(errorEntry.id, { status: 'fix_failed', note: `AI request failed: ${err.message}` });
  }

  if (result.dryRun || !result.fixedFile) {
    return errors.update(errorEntry.id, { status: 'unresolved', note: result.summary });
  }

  fs.writeFileSync(targetFile, result.fixedFile);

  const testResult = await test();
  if (!testResult.passed) {
    fs.writeFileSync(targetFile, original); // revert - a broken fix never gets committed
    return errors.update(errorEntry.id, {
      status: 'fix_failed',
      note: `Candidate fix failed tests: ${testResult.output || 'unknown failure'}`,
    });
  }

  const relFile = path.relative(config.repoRoot, targetFile);
  const deployResult = await deployer.commitAndDeploy({
    files: [relFile],
    message: `fix(auto): ${result.summary || 'AI remediation'} [error ${errorEntry.id}]`,
  });

  return errors.update(errorEntry.id, {
    status: 'fixed',
    note: result.summary,
    deploy: deployResult,
  });
}

module.exports = { remediate, resolveTargetFile, runTests };
