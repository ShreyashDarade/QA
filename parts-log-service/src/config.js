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

  // Deploy targets. A candidate fix is pushed to devBranch first, live-
  // validated there (see remediationLoop.js), and only pushed on to
  // prodBranch once validation passes. Both are just git branches by
  // default (safe in any repo) - point DEPLOY_WEBHOOK_URL / the
  // deploy-prod workflow / K8S_* at real infra when you're ready.
  devBranch: process.env.DEV_BRANCH || 'dev',
  prodBranch: process.env.PROD_BRANCH || 'prod',
  gitRemote: process.env.GIT_REMOTE || 'origin',
  deployWebhookUrl: process.env.DEPLOY_WEBHOOK_URL || '',

  // Command run as a fast local gate before a fix is even pushed to dev.
  // Default runs lint + format check + the full test suite, so a candidate
  // fix that's stylistically inconsistent or that regresses any existing
  // test never gets past this gate.
  testCommand: process.env.TEST_COMMAND || 'npm run verify',

  // The retrieve-analyze-fix-push-restart-test-monitor loop
  // (remediationLoop.js) repeats up to this many times per error before
  // giving up - this is the loop bound for "continue this loop until all
  // APIs pass successfully."
  maxRemediationCycles: Number(process.env.MAX_REMEDIATION_CYCLES) || 5,

  // Kubernetes restart after pushing a fix to devBranch. Unconfigured
  // (no K8S_DEPLOYMENT) means this step is skipped, not failed - the loop
  // still runs, it just doesn't restart anything.
  k8sNamespace: process.env.K8S_NAMESPACE || 'default',
  k8sDeployment: process.env.K8S_DEPLOYMENT || '',
  kubectlPath: process.env.KUBECTL_PATH || 'kubectl',
  k8sRolloutTimeout: process.env.K8S_ROLLOUT_TIMEOUT || '120s',

  // Postman collection run against the redeployed service to confirm the
  // affected APIs now succeed. Unconfigured (no POSTMAN_COLLECTION_PATH)
  // means this step is skipped and treated as passing, not failed.
  postmanCollectionPath: process.env.POSTMAN_COLLECTION_PATH || '',
  postmanBaseUrl:
    process.env.POSTMAN_BASE_URL || `http://localhost:${Number(process.env.PORT) || 3000}`,

  // How long to watch application logs for new errors: once right after
  // each test cycle (short), and once more after a fix is promoted to prod
  // to confirm the service stays stable (longer).
  monitorWindowMs: Number(process.env.MONITOR_WINDOW_MS) || 30000,
  monitorPollIntervalMs: Number(process.env.MONITOR_POLL_INTERVAL_MS) || 3000,
  postDeployMonitorWindowMs: Number(process.env.POST_DEPLOY_MONITOR_WINDOW_MS) || 60000,
};
