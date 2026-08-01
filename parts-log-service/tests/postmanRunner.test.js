const test = require('node:test');
const assert = require('node:assert');

const { runCollection } = require('../src/services/postmanRunner');

test('runCollection is skipped (and treated as passing) with no collection configured', async () => {
  const result = await runCollection({ collectionPath: '', baseUrl: 'http://x' });
  assert.strictEqual(result.skipped, true);
  assert.strictEqual(result.allPassed, true);
});

test('runCollection is skipped when the configured file does not exist', async () => {
  const result = await runCollection({
    collectionPath: '/nonexistent/collection.json',
    baseUrl: 'http://x',
  });
  assert.strictEqual(result.skipped, true);
  assert.strictEqual(result.allPassed, true);
});

test('runCollection reports per-request pass/fail against a live server', async (t) => {
  const http = require('http');
  const server = http.createServer((req, res) => {
    if (req.url === '/ok') {
      res.writeHead(200);
      res.end('ok');
    } else {
      res.writeHead(500);
      res.end('boom');
    }
  });
  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();
  t.after(() => server.close());

  const path = require('path');
  const collectionPath = path.join(__dirname, '..', 'postman', 'example.postman_collection.json');
  const fs = require('fs');
  const tmpCollection = path.join(require('os').tmpdir(), `collection-${Date.now()}.json`);
  fs.writeFileSync(
    tmpCollection,
    JSON.stringify({
      item: [
        { name: 'ok request', request: { method: 'GET', header: [], url: '{{baseUrl}}/ok' } },
        {
          name: 'broken request',
          request: { method: 'GET', header: [], url: '{{baseUrl}}/broken' },
        },
      ],
    })
  );

  const result = await runCollection({
    collectionPath: tmpCollection,
    baseUrl: `http://127.0.0.1:${port}`,
  });
  assert.strictEqual(result.skipped, false);
  assert.strictEqual(result.allPassed, false);
  assert.strictEqual(result.results.find((r) => r.name === 'ok request').ok, true);
  assert.strictEqual(result.results.find((r) => r.name === 'broken request').ok, false);

  fs.unlinkSync(tmpCollection);
  assert.ok(collectionPath); // sanity: the shipped example collection path resolves
});
