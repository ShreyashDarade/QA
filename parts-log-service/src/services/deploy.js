const simpleGit = require('simple-git');
const config = require('../config');

/**
 * Commits the currently staged/working-tree changes and pushes to the given
 * branch (devBranch by default) - no human approval step, per design. The
 * only gates are automated: the local test suite (see aiFixer.js) before
 * this is even called, and live validation in remediationLoop.js before
 * promoteToProd() is ever called. By default this just pushes a git branch,
 * which is safe against any repo - point PROD_BRANCH / DEV_BRANCH /
 * DEPLOY_WEBHOOK_URL / K8S_* at real infra when you're ready.
 */
async function commitAndDeploy({ files, message, branch }) {
  const git = simpleGit(config.repoRoot);
  const targetBranch = branch || config.devBranch;

  await git.add(files);
  await git.commit(message);
  await git.push(config.gitRemote, `HEAD:${targetBranch}`, ['--force-with-lease']);

  await callDeployWebhook();

  const log = await git.log({ maxCount: 1 });
  return { commit: log.latest ? log.latest.hash : null, branch: targetBranch };
}

/**
 * Promotes the currently checked-out commit (the one that just passed live
 * validation on devBranch) to prodBranch. Left as a separate step so it is
 * only ever called after remediationLoop.js has confirmed the affected
 * APIs pass and the logs stayed clean.
 */
async function promoteToProd() {
  const git = simpleGit(config.repoRoot);
  await git.push(config.gitRemote, `HEAD:${config.prodBranch}`, ['--force-with-lease']);

  await callDeployWebhook();

  const log = await git.log({ maxCount: 1 });
  return { commit: log.latest ? log.latest.hash : null, branch: config.prodBranch };
}

async function callDeployWebhook() {
  if (!config.deployWebhookUrl) return;
  await fetch(config.deployWebhookUrl, { method: 'POST' }).catch((err) => {
    console.error('[deploy] webhook call failed:', err.message);
  });
}

module.exports = { commitAndDeploy, promoteToProd };
