const test = require('node:test');
const assert = require('node:assert');

process.env.K8S_DEPLOYMENT = '';

const { restartAndWait } = require('../src/services/k8s');

test('restartAndWait skips (not fails) when no deployment is configured', async () => {
  const result = await restartAndWait();
  assert.strictEqual(result.skipped, true);
});
