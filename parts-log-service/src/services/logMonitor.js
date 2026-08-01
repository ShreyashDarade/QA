const { errors } = require('../middleware/errorLogger');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Polls the same error log every other part of this service writes to
 * (in-process captures + externally reported errors via POST /api/errors)
 * for a window of time, looking for anything new for the given part. Used
 * both right after a test cycle (short window) and after a fix is promoted
 * to prod (longer window, "continue monitoring... to ensure the service
 * remains stable").
 *
 * Returns as soon as something new shows up, rather than always waiting
 * out the full window - a fresh error is a clear "not stable" signal, no
 * need to keep polling once we have it.
 */
async function watch({ partKey, sinceIso, windowMs, pollIntervalMs, excludeErrorId }) {
  const since = new Date(sinceIso);
  const deadline = Date.now() + windowMs;

  while (Date.now() < deadline) {
    const fresh = errors.find(
      (e) =>
        e.id !== excludeErrorId &&
        (!partKey || e.partKey === partKey) &&
        new Date(e.createdAt) > since
    );
    if (fresh.length) {
      return { clean: false, newErrors: fresh };
    }
    await sleep(Math.max(0, Math.min(pollIntervalMs, deadline - Date.now())));
  }

  return { clean: true, newErrors: [] };
}

module.exports = { watch };
