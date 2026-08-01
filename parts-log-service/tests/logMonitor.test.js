const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

process.env.DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'parts-log-monitor-test-'));

const { errors } = require('../src/middleware/errorLogger');
const { watch } = require('../src/services/logMonitor');

test('watch resolves clean when no new errors appear in the window', async () => {
  const sinceIso = new Date().toISOString();
  const result = await watch({ partKey: 'p1', sinceIso, windowMs: 150, pollIntervalMs: 20 });
  assert.strictEqual(result.clean, true);
  assert.deepStrictEqual(result.newErrors, []);
});

test('watch detects a new error for the part and returns early', async () => {
  const sinceIso = new Date().toISOString();

  setTimeout(() => {
    errors.insert({
      id: 'watch-err-1',
      partKey: 'p2',
      message: 'boom',
      stack: null,
      source: 'test',
      metadata: {},
      status: 'new',
      fixAttempts: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
  }, 20);

  const start = Date.now();
  const result = await watch({ partKey: 'p2', sinceIso, windowMs: 2000, pollIntervalMs: 20 });
  const elapsed = Date.now() - start;

  assert.strictEqual(result.clean, false);
  assert.strictEqual(result.newErrors.length, 1);
  assert.ok(elapsed < 2000, 'should return as soon as a new error is found, not wait out the full window');
});

test('watch ignores the excluded error id and errors for other parts', async () => {
  const sinceIso = new Date().toISOString();
  errors.insert({
    id: 'watch-err-self',
    partKey: 'p3',
    message: 'the error being remediated itself',
    stack: null,
    source: 'test',
    metadata: {},
    status: 'fixing',
    fixAttempts: 1,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
  errors.insert({
    id: 'watch-err-other-part',
    partKey: 'other-part',
    message: 'unrelated',
    stack: null,
    source: 'test',
    metadata: {},
    status: 'new',
    fixAttempts: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  const result = await watch({
    partKey: 'p3',
    sinceIso,
    windowMs: 150,
    pollIntervalMs: 20,
    excludeErrorId: 'watch-err-self',
  });
  assert.strictEqual(result.clean, true);
});
