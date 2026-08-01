const express = require('express');
const config = require('./config');

const { router: registryRouter } = require('./routes/registry');
const { router: partsRouter } = require('./routes/parts');
const { router: errorsRouter } = require('./routes/errors');
const { errorLogger } = require('./middleware/errorLogger');
const { remediate } = require('./services/remediationLoop');

function createApp() {
  const app = express();
  app.use(express.json());

  app.get('/health', (req, res) => res.json({ ok: true }));

  app.use('/api/registry', registryRouter);
  app.use('/api/parts', partsRouter);
  app.use('/api/errors', errorsRouter);

  app.use((req, res) => res.status(404).json({ error: 'not_found' }));

  // Deterministic, non-AI capture middleware goes last. Once an error is
  // captured, it is handed off to the retrieve/analyze/fix/deploy/validate
  // remediation loop (see services/remediationLoop.js) as a fully separate,
  // asynchronous step.
  app.use(errorLogger(remediate));

  return app;
}

if (require.main === module) {
  const app = createApp();
  app.listen(config.port, () => {
    // eslint-disable-next-line no-console
    console.log(`parts-log-service listening on :${config.port}`);
  });
}

module.exports = { createApp };
