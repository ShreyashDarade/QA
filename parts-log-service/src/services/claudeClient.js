const config = require('../config');

const FILE_TAG = '@@FIXED_FILE@@';
const SUMMARY_TAG = '@@SUMMARY@@';

/**
 * Thin wrapper around the Anthropic SDK. Isolated in its own module so the
 * rest of the AI fixer can be tested without a live API key (see
 * config.aiFixerDryRun / the injected client in aiFixer tests).
 */
function buildPrompt({ errorEntry, fileName, fileContents }) {
  return `You are an automated remediation agent for a software "parts" logging service.
An error was captured by deterministic (non-AI) error-handling middleware. Your job is to
propose the smallest safe fix to the single file below that resolves the error, without
changing unrelated behavior.

Error message: ${errorEntry.message}
Stack trace:
${errorEntry.stack || '(none provided)'}

File: ${fileName}
--- BEGIN FILE ---
${fileContents}
--- END FILE ---

Respond with EXACTLY two sections, in this order, and nothing else:

${SUMMARY_TAG}
<one or two sentence explanation of the root cause and the fix>

${FILE_TAG}
<the COMPLETE corrected contents of the file, ready to write to disk verbatim>
`;
}

function parseResponse(text) {
  const fileIdx = text.indexOf(FILE_TAG);
  const summaryIdx = text.indexOf(SUMMARY_TAG);
  if (fileIdx === -1 || summaryIdx === -1) {
    throw new Error('AI response did not contain the expected sections');
  }
  const summary = text.slice(summaryIdx + SUMMARY_TAG.length, fileIdx).trim();
  const fixedFile = text.slice(fileIdx + FILE_TAG.length).trim();
  return { summary, fixedFile };
}

async function requestFix({ errorEntry, fileName, fileContents }) {
  if (!config.anthropicApiKey || config.aiFixerDryRun) {
    return {
      dryRun: true,
      summary: 'Dry-run mode (no ANTHROPIC_API_KEY, or AI_FIXER_DRY_RUN=true): no fix generated.',
      fixedFile: null,
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
