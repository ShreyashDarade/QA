require('dotenv').config();

function bool(value, fallback) {
  if (value === undefined) return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(value).toLowerCase());
}

module.exports = {
  port: Number(process.env.PORT) || 3000,
  dataDir: process.env.DATA_DIR || require('path').join(__dirname, '..', 'data'),

  // Anthropic / AI remediation
  anthropicApiKey: process.env.ANTHROPIC_API_KEY || '',
  anthropicModel: process.env.ANTHROPIC_MODEL || 'claude-sonnet-5',
  aiFixerEnabled: bool(process.env.AI_FIXER_ENABLED, true),
  // When no API key is configured, the fixer runs in dry-run mode: it still
  // captures and analyzes errors, but never fabricates or applies a patch.
  aiFixerDryRun: bool(process.env.AI_FIXER_DRY_RUN, false),

  // Repo root the AI fixer is allowed to patch. Defaults to this service's
  // own repo so a fresh checkout works with zero configuration.
  repoRoot: process.env.REPO_ROOT || require('path').join(__dirname, '..'),

  // Deploy target: which branch a validated fix is pushed to, and which
  // remote. This is intentionally just a git branch by default (safe in any
  // repo) - point DEPLOY_WEBHOOK_URL / the deploy-prod workflow at real
  // infra when you're ready to wire up an actual production target.
  prodBranch: process.env.PROD_BRANCH || 'prod',
  gitRemote: process.env.GIT_REMOTE || 'origin',
  deployWebhookUrl: process.env.DEPLOY_WEBHOOK_URL || '',

  // Command run to validate a candidate fix before it is ever pushed.
  testCommand: process.env.TEST_COMMAND || 'npm test',
};
