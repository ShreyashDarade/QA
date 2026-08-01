const { execFile } = require('child_process');
const { promisify } = require('util');
const config = require('../config');

const execFileAsync = promisify(execFile);

/**
 * Restarts the configured Kubernetes deployment and waits for the rollout
 * to finish, so the freshly pushed dev-branch fix is actually live before
 * the Postman step exercises it. Unconfigured (no K8S_DEPLOYMENT) means
 * this is skipped - not failed - so the rest of the loop still runs
 * end-to-end without requiring a cluster.
 */
async function restartAndWait() {
  if (!config.k8sDeployment) {
    return { skipped: true, note: 'K8S_DEPLOYMENT not set - restart skipped.' };
  }

  const nsArgs = ['-n', config.k8sNamespace];
  const target = `deployment/${config.k8sDeployment}`;

  try {
    await execFileAsync(config.kubectlPath, ['rollout', 'restart', target, ...nsArgs]);
    await execFileAsync(config.kubectlPath, [
      'rollout',
      'status',
      target,
      ...nsArgs,
      `--timeout=${config.k8sRolloutTimeout}`,
    ]);
    return { skipped: false, restarted: true };
  } catch (err) {
    return { skipped: false, restarted: false, error: err.message };
  }
}

module.exports = { restartAndWait };
