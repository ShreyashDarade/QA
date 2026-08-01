const test = require('node:test');
const assert = require('node:assert');

const { parseResponse } = require('../src/services/claudeClient');

test('parseResponse extracts summary and fixed file from a well-formed response', () => {
  const text = `@@SUMMARY@@\nFixed the off-by-one error.\n\n@@FIXED_FILE@@\nconsole.log("hi");\n`;
  const { summary, fixedFile } = parseResponse(text);
  assert.strictEqual(summary, 'Fixed the off-by-one error.');
  assert.strictEqual(fixedFile, 'console.log("hi");');
});

test('parseResponse throws on malformed responses', () => {
  assert.throws(() => parseResponse('nonsense response with no tags'));
});
