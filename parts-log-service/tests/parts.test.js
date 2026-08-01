const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

process.env.DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'parts-log-test-'));
process.env.AI_FIXER_DRY_RUN = 'true';

const { createApp } = require('../src/server');

async function request(app, method, url, body) {
  return new Promise((resolve, reject) => {
    const server = app.listen(0, () => {
      const { port } = server.address();
      fetch(`http://127.0.0.1:${port}${url}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      })
        .then(async (res) => {
          const json = await res.json().catch(() => null);
          server.close();
          resolve({ status: res.status, body: json });
        })
        .catch((err) => {
          server.close();
          reject(err);
        });
    });
  });
}

test('POST /api/parts requires partKey and event', async () => {
  const app = createApp();
  const res = await request(app, 'POST', '/api/parts', { event: 'installed' });
  assert.strictEqual(res.status, 400);
});

test('parts log entries are addable externally with an arbitrary partKey', async () => {
  const app = createApp();
  const created = await request(app, 'POST', '/api/parts', {
    partKey: 'checkout-service',
    event: 'deployed',
    level: 'info',
    metadata: { version: '1.2.3' },
  });
  assert.strictEqual(created.status, 201);
  assert.strictEqual(created.body.partKey, 'checkout-service');
  assert.strictEqual(created.body.registeredKey, null); // not registered, still logged

  const list = await request(app, 'GET', '/api/parts?partKey=checkout-service');
  assert.strictEqual(list.status, 200);
  assert.strictEqual(list.body.length, 1);
});

test('registry accepts new part keys/aliases with no fixed schema', async () => {
  const app = createApp();
  const created = await request(app, 'POST', '/api/registry', {
    key: 'checkout-service',
    file: 'src/checkout.js',
    aliases: ['co', 'checkout'],
  });
  assert.strictEqual(created.status, 201);

  const byAlias = await request(app, 'GET', '/api/registry/co');
  assert.strictEqual(byAlias.status, 200);
  assert.strictEqual(byAlias.body.key, 'checkout-service');
});

test('unhandled route errors are captured deterministically without AI', async () => {
  const express = require('express');
  const { errorLogger } = require('../src/middleware/errorLogger');

  const app = express();
  app.use(express.json());
  app.get('/__boom', () => {
    throw new Error('boom');
  });
  app.use(errorLogger());

  const res = await request(app, 'GET', '/__boom');
  assert.strictEqual(res.status, 500);
  assert.strictEqual(res.body.error, 'internal_error');
  assert.ok(res.body.errorId);

  const errList = await request(createApp(), 'GET', '/api/errors');
  assert.strictEqual(errList.status, 200);
  assert.ok(errList.body.some((e) => e.id === res.body.errorId));
});
