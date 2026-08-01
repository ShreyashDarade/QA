const config = require('../config');

const FILE_TAG = '@@FIXED_FILE@@';
const SUMMARY_TAG = '@@SUMMARY@@';
const TEST_PATH_TAG = '@@TEST_FILE_PATH@@';
const TEST_FILE_TAG = '@@TEST_FILE@@';

/**
 * Thin wrapper around the Anthropic SDK. Isolated in its own module so the
 * rest of the AI fixer can be tested without a live API key (see
 * config.aiFixerDryRun / the injected client in aiFixer tests).
 */
function buildPrompt({ errorEntry, fileName, fileContents }) {
  return `You are an automated remediation agent for a software "parts" logging service.
An error was captured by deterministic (non-AI) error-handling middleware. Your job is to
propose the smallest correct fix to the single file below that resolves the root cause, without
changing unrelated behavior.

Rules:
- Prefer the smallest correct fix. Refactor only what's necessary to fix the bug correctly.
- Match the file's existing naming conventions and formatting style; the result will be run
  through a linter and formatter, so it must already be consistent with the rest of the file.
- Any NEW helper functions or variables you introduce should have clear, descriptive names
  consistent with the rest of the file.
- Do NOT rename anything this file exports via module.exports, and do not change its public
  function signatures, unless the rename/signature change is itself required to fix the bug -
  other files that import this one are not shown to you, and an unnecessary rename here could
  silently break them with no one reviewing the change before it ships.
- If the repository's existing test suite does not already cover this bug, also propose a new or
  updated test file that would have caught it. If existing tests already cover this case, omit
  the test sections entirely rather than duplicating coverage.

Error message: ${errorEntry.message}
Stack trace:
${errorEntry.stack || '(none provided)'}

File: ${fileName}
--- BEGIN FILE ---
${fileContents}
--- END FILE ---

Respond with EXACTLY these sections, in this order, and nothing else. Omit the two TEST sections
together if no test addition/update is warranted.

${SUMMARY_TAG}
<one or two sentence explanation of the root cause and the fix>

${FILE_TAG}
<the COMPLETE corrected contents of the file, ready to write to disk verbatim>

${TEST_PATH_TAG} (optional)
<repo-relative path of a new or existing test file to add/update coverage for this bug>

${TEST_FILE_TAG} (optional)
<the COMPLETE contents of that test file, ready to write to disk verbatim>
`;
}

function parseResponse(text) {
  const summaryIdx = text.indexOf(SUMMARY_TAG);
  const fileIdx = text.indexOf(FILE_TAG);
  if (fileIdx === -1 || summaryIdx === -1) {
    throw new Error('AI response did not contain the expected sections');
  }

  const testPathIdx = text.indexOf(TEST_PATH_TAG);
  const testFileIdx = text.indexOf(TEST_FILE_TAG);
  const hasTestSections = testPathIdx !== -1 && testFileIdx !== -1;

  const summary = text.slice(summaryIdx + SUMMARY_TAG.length, fileIdx).trim();
  const fixedFile = text
    .slice(fileIdx + FILE_TAG.length, hasTestSections ? testPathIdx : undefined)
    .trim();

  if (!hasTestSections) {
    return { summary, fixedFile, testFilePath: null, testFile: null };
  }

  const testFilePath = text.slice(testPathIdx + TEST_PATH_TAG.length, testFileIdx).trim();
  const testFile = text.slice(testFileIdx + TEST_FILE_TAG.length).trim();
  return { summary, fixedFile, testFilePath: testFilePath || null, testFile: testFile || null };
}

async function requestFix({ errorEntry, fileName, fileContents }) {
  if (!config.anthropicApiKey || config.aiFixerDryRun) {
    return {
      dryRun: true,
      summary: 'Dry-run mode (no ANTHROPIC_API_KEY, or AI_FIXER_DRY_RUN=true): no fix generated.',
      fixedFile: null,
      testFilePath: null,
      testFile: null,
    };
  }

  // Lazy-require so the SDK is only needed when a real call is made.
  const Anthropic = require('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey: config.anthropicApiKey });

  const prompt = buildPrompt({ errorEntry, fileName, fileContents });

  const response = await client.messages.create({
    model: config.anthropicModel,
    max_tokens: 4096,
    messages: [{ role: 'user', content: prompt }],
  });

  const text = response.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('\n');

  return { dryRun: false, ...parseResponse(text) };
}

module.exports = { requestFix, buildPrompt, parseResponse };
