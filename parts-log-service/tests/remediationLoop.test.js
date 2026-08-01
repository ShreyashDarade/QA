const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpRepo = fs.mkdtempSync(path.join(os.tmpdir(), 'parts-log-remloop-'));
process.env.DATA_DIR = path.join(tmpRepo, 'data');
process.env.REPO_ROOT = tmpRepo;
process.env.AI_FIXER_DRY_RUN = 'true';

const targetFile = path.join(tmpRepo, 'buggy.js');
fs.writeFileSync(targetFile, 'function add(a, b) { return a - b; }\n');

const config = require('../src/config');
const { errors } = require('../src/middleware/errorLogger');
const { remediate } = require('../src/services/remediationLoop');

function makeError(id, partKey) {
  return errors.insert({
    id,
    partKey,
    message: 'add() returns the wrong value',
    stack: null,
    source: 'unit-test',
    metadata: {},
    status: 'new',
    fixAttempts: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
}

const passingDeps = () => ({
  resolveTargetFile: () => targetFile,
  runTests: async () => ({ passed: true }),
  claudeClient: { requestFix: async () => ({ dryRun: false, summary: 'fixed it', fixedFile: 'function add(a, b) { return a + b; }\n' }) },
  deploy: {
    commitAndDeploy: async () => ({ commit: 'dev123', branch: 'dev' }),
    promoteToProd: async () => ({ commit: 'prod123', branch: 'prod' }),
  },
  k8s: { restartAndWait: async () => ({ skipped: true }) },
  postmanRunner: { runCollection: async () => ({ skipped: true, allPassed: true, results: [] }) },
  logMonitor: { watch: async () => ({ clean: true, newErrors: [] }) },
});

test('remediate: validates and promotes to prod on the first cycle when everything passes', async () => {
  const errorEntry = makeError('rl-1', 'math-part');
  const result = await remediate(errorEntry, passingDeps());

  assert.strictEqual(result.status, 'fixed');
  assert.strictEqual(result.deploy.dev.branch, 'dev');
  assert.strictEqual(result.deploy.prod.branch, 'prod');
  assert.strictEqual(result.history.length, 1);
  assert.strictEqual(result.stability.clean, true);
});

test('remediate: retries after a failing Postman run, then succeeds and promotes', async () => {
  const errorEntry = makeError('rl-2', 'math-part');
  let postmanCalls = 0;

  const deps = passingDeps();
  deps.postmanRunner = {
    runCollection: async () => {
      postmanCalls += 1;
      if (postmanCalls === 1) {
        return { skipped: false, allPassed: false, results: [{ name: 'add endpoint', method: 'GET', url: 'x', status: 500, ok: false }] };
      }
      return { skipped: false, allPassed: true, results: [{ name: 'add endpoint', method: 'GET', url: 'x', status: 200, ok: true }] };
    },
  };

  const result = await remediate(errorEntry, deps);

  assert.strictEqual(result.status, 'fixed');
  assert.strictEqual(postmanCalls, 2);
  assert.strictEqual(result.history.length, 2);
  assert.strictEqual(result.history[0].outcome, 'revalidation_failed');
  assert.strictEqual(result.history[1].outcome, 'validated');
});

test('remediate: gives up after MAX_REMEDIATION_CYCLES and never promotes', async () => {
  const originalMax = config.maxRemediationCycles;
  config.maxRemediationCycles = 2;
  try {
    const errorEntry = makeError('rl-3', 'math-part');
    let promoted = false;

    const deps = passingDeps();
    deps.postmanRunner = { runCollection: async () => ({ skipped: false, allPassed: false, results: [] }) };
    deps.deploy.promoteToProd = async () => {
      promoted = true;
      return {};
    };

    const result = await remediate(errorEntry, deps);

    assert.strictEqual(result.status, 'fix_failed');
    assert.strictEqual(result.history.length, 2);
    assert.strictEqual(promoted, false);
  } finally {
    config.maxRemediationCycles = originalMax;
  }
});

test('remediate: marks unresolved immediately when no source file is registered for the part', async () => {
  const errorEntry = makeError('rl-4', 'unregistered-part');
  const deps = passingDeps();
  deps.resolveTargetFile = () => null;

  const result = await remediate(errorEntry, deps);

  assert.strictEqual(result.status, 'unresolved');
  assert.strictEqual(result.history.length, 1);
});
