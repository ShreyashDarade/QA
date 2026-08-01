const { v4: uuid } = require('uuid');
const { Collection } = require('../store');

const errors = new Collection('errors');

/**
 * Plain, deterministic error capture - NO AI involved here by design.
 * Any error thrown (or passed to next(err)) by a route handler lands here,
 * gets a structured record written to the errors log, and the response is
 * still answered safely. AI only enters the picture afterwards, as a
 * separate consumer of this log (see services/aiFixer.js).
 */
function errorLogger(onCaptured) {
  // eslint-disable-next-line no-unused-vars
  return function handleError(err, req, res, next) {
    const entry = {
      id: uuid(),
      partKey: (req.body && req.body.partKey) || req.query.partKey || null,
      message: err.message || String(err),
      stack: err.stack || null,
      source: `${req.method} ${req.originalUrl}`,
      metadata: {},
      status: 'new',
      fixAttempts: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    errors.insert(entry);

    if (typeof onCaptured === 'function') {
      // Fire-and-forget: handing the captured error to the AI remediation
      // pipeline must never block or fail the HTTP response.
      Promise.resolve()
        .then(() => onCaptured(entry))
        .catch((fixerErr) => {
          // eslint-disable-next-line no-console
          console.error('[errorLogger] AI fixer invocation failed:', fixerErr.message);
        });
    }

    const status = err.status || err.statusCode || 500;
    res.status(status).json({
      error: 'internal_error',
      message: 'The error was captured and logged.',
      errorId: entry.id,
    });
  };
}

module.exports = { errorLogger, errors };
