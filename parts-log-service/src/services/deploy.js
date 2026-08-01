const simpleGit = require('simple-git');
const config = require('../config');

/**
 * Commits the currently staged/working-tree changes and pushes straight to
 * the configured prod branch - no human approval step, per design. The only
 * gate is the caller having already run the test suite successfully
 * (see services/aiFixer.js). Point PROD_BRANCH / GIT_REMOTE /
 * DEPLOY_WEBHOOK_URL at real infra when you're ready; by default this just
 * pushes a git branch, which is safe against any repo.
 */
async function commitAndDeploy({ files, message }) {
  const git = simpleGit(config.repoRoot);

  await git.add(files);
  await git.commit(message);
  await git.push(config.gitRemote, `HEAD:${config.prodBranch}`, ['--force-with-lease']);

  if (config.deployWebhookUrl) {
    await fetch(config.deployWebhookUrl, { method: 'POST' }).catch((err) => {
      // eslint-disable-next-line no-console
      console.error('[deploy] webhook call failed:', err.message);
    });
  }

  const log = await git.log({ maxCount: 1 });
  return { commit: log.latest ? log.latest.hash : null, branch: config.prodBranch };
}

module.exports = { commitAndDeploy };
