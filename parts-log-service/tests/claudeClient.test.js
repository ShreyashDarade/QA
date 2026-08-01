const test = require('node:test');
const assert = require('node:assert');

const { parseResponse } = require('../src/services/claudeClient');

test('parseResponse extracts summary and fixed file from a well-formed response', () => {
  const text = `@@SUMMARY@@\nFixed the off-by-one error.\n\n@@FIXED_FILE@@\nconsole.log("hi");\n`;
  const { summary, fixedFile, testFilePath, testFile } = parseResponse(text);
  assert.strictEqual(summary, 'Fixed the off-by-one error.');
  assert.strictEqual(fixedFile, 'console.log("hi");');
  assert.strictEqual(testFilePath, null);
  assert.strictEqual(testFile, null);
});

test('parseResponse extracts an optional companion test file when present', () => {
  const text = [
    '@@SUMMARY@@',
    'Fixed the off-by-one error.',
    '',
    '@@FIXED_FILE@@',
    'console.log("hi");',
    '',
    '@@TEST_FILE_PATH@@',
    'tests/greeting.test.js',
    '',
    '@@TEST_FILE@@',
    "test('greets', () => {});",
  ].join('\n');

  const { testFilePath, testFile } = parseResponse(text);
  assert.strictEqual(testFilePath, 'tests/greeting.test.js');
  assert.strictEqual(testFile, "test('greets', () => {});");
});

test('parseResponse throws on malformed responses', () => {
  assert.throws(() => parseResponse('nonsense response with no tags'));
});
