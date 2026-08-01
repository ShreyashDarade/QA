const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpRepo = fs.mkdtempSync(path.join(os.tmpdir(), 'parts-log-aifixer-'));
process.env.DATA_DIR = path.join(tmpRepo, 'data');
process.env.REPO_ROOT = tmpRepo;
process.env.AI_FIXER_DRY_RUN = 'true';

const targetFile = path.join(tmpRepo, 'buggy.js');
fs.writeFileSync(targetFile, 'function add(a, b) { return a - b; }\nmodule.exports = { add };\n');

const { registry } = require('../src/routes/registry');
const { errors } = require('../src/middleware/errorLogger');
const { remediate } = require('../src/services/aiFixer');

registry.insert({
  id: 'reg-1',
  key: 'math-part',
  file: 'buggy.js',
  description: 'test part',
  aliases: [],
  metadata: {},
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
});

test('remediate applies an AI fix, runs tests, and deploys on success', async () => {
  const errorEntry = errors.insert({
    id: 'err-1',
    partKey: 'math-part',
    message: 'add(2, 3) returned -1, expected 5',
    stack: null,
    source: 'unit-test',
    metadata: {},
    status: 'new',
    fixAttempts: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  const fixedFile = 'function add(a, b) { return a + b; }\nmodule.exports = { add };\n';
  let deployCalled = null;

  const result = await remediate(errorEntry, {
    claudeClient: {
      requestFix: async () => ({ dryRun: false, summary: 'Fixed subtraction to addition', fixedFile }),
    },
    runTests: async () => ({ passed: true }),
    deploy: {
      commitAndDeploy: async (args) => {
        deployCalled = args;
        return { commit: 'abc123', branch: 'prod' };
      },
    },
  });

  assert.strictEqual(result.status, 'fixed');
  assert.strictEqual(fs.readFileSync(targetFile, 'utf8'), fixedFile);
  assert.ok(deployCalled);
  assert.deepStrictEqual(deployCalled.files, ['buggy.js']);
});

test('remediate reverts the file and does not deploy when tests fail', async () => {
  const original = fs.readFileSync(targetFile, 'utf8');
  const errorEntry = errors.insert({
    id: 'err-2',
    partKey: 'math-part',
    message: 'still broken',
    stack: null,
    source: 'unit-test',
    metadata: {},
    status: 'new',
    fixAttempts: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  let deployCalled = false;

  const result = await remediate(errorEntry, {
    claudeClient: {
      requestFix: async () => ({ dryRun: false, summary: 'bad fix', fixedFile: 'this is not valid js' }),
    },
    runTests: async () => ({ passed: false, output: 'SyntaxError' }),
    deploy: {
      commitAndDeploy: async () => {
        deployCalled = true;
        return {};
      },
    },
  });

  assert.strictEqual(result.status, 'fix_failed');
  assert.strictEqual(fs.readFileSync(targetFile, 'utf8'), original);
  assert.strictEqual(deployCalled, false);
});

test('remediate marks unresolved when the part has no registered file', async () => {
  const errorEntry = errors.insert({
    id: 'err-3',
    partKey: 'unknown-part',
    message: 'no target',
    stack: null,
    source: 'unit-test',
    metadata: {},
    status: 'new',
    fixAttempts: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  const result = await remediate(errorEntry, {
    claudeClient: { requestFix: async () => { throw new Error('should not be called'); } },
  });

  assert.strictEqual(result.status, 'unresolved');
});
