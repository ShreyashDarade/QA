# parts-log-service

A universal activity log for software "parts" (services, modules, jobs — any
named unit of your system), with a deterministic error catcher and an
autonomous AI remediation pipeline sitting behind it.

## How it fits together

```
external caller ──▶ POST /api/parts          (log an event for any part, addable at runtime)
external caller ──▶ POST /api/registry        (register/describe a part: its source file, aliases, metadata)
external caller ──▶ POST /api/errors          (report an error from outside this process)
in-process error ──▶ errorLogger middleware   (plain try/catch style capture - NO AI here)
                          │
                          ▼
                    services/aiFixer.js        (only place AI is invoked)
                          │  1. look up the part's source file in the registry
                          │  2. ask Claude for a minimal fix (full corrected file)
                          │  3. write the fix, run the test suite
                          │  4. tests pass  -> commit + push straight to PROD_BRANCH (no human step)
                          │     tests fail  -> revert the file, mark fix_failed
                          ▼
                    services/deploy.js          (git push to prod branch, optional deploy webhook)
                          │
                          ▼
                 .github/workflows/deploy-prod.yml  (fires on push to prod, re-runs tests, deploys)
```

Nothing here is hardcoded to a specific "part": what a part is, what file
backs it, and what aliases/key-commands refer to it are all supplied through
the API at runtime (`POST /api/registry`, `POST /api/parts`). Add a new part
by registering it - no code change required.

## Why the AI only sees captured errors, never raw requests

`errorLogger` is intentionally dumb: it's an Express error-handling
middleware that logs whatever exception reached it, with no branching logic
or model calls. AI only enters the picture afterwards, as a separate
consumer reading from the same error log (`services/aiFixer.js`). That
keeps error *detection* deterministic and testable, and confines the AI's
blast radius to *remediation* of already-captured, already-logged issues.

## Autonomy and safety rails

Per project requirements this pipeline is **fully autonomous**: a validated
fix is pushed to `PROD_BRANCH` with no human approval step. The only gate is
automated: the candidate fix must pass `TEST_COMMAND` (default `npm test`)
before it is ever committed. If tests fail, the file is reverted in place
and nothing is pushed. Review `src/services/aiFixer.js` and
`src/services/deploy.js` before pointing `DEPLOY_WEBHOOK_URL` /
`.github/workflows/deploy-prod.yml` at a real production target - the
default behavior only pushes a git branch, which is safe against any repo,
but wiring it to real infrastructure raises the stakes of a bad AI fix
considerably. Consider branch protection / required status checks on
`PROD_BRANCH` if you want CI to be a second automated gate even though
there's no human one.

## API

### `POST /api/registry`
Register or update a part. `key` is the only required field.
```json
{ "key": "checkout-service", "file": "src/checkout.js", "aliases": ["co"], "metadata": { "owner": "payments" } }
```

### `GET /api/registry` / `GET /api/registry/:key`
List all registered parts, or look one up by key or alias.

### `POST /api/parts`
Log an event for any part. `partKey` does not need to be pre-registered.
```json
{ "partKey": "checkout-service", "event": "deployed", "level": "info", "metadata": { "version": "1.2.3" } }
```

### `GET /api/parts?partKey=&level=&since=`
Query the parts log.

### `POST /api/errors`
Report an error from outside this process (e.g. another service, a log
shipper). Lands in the same pipeline as errors caught in-process.
```json
{ "partKey": "checkout-service", "message": "TypeError: cannot read x of undefined", "stack": "..." }
```

### `GET /api/errors?status=&partKey=`
Inspect the error/remediation audit trail. `status` is one of `new`,
`fixing`, `fixed`, `fix_failed`, `unresolved`, `skipped`.

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables real AI remediation. Unset = dry-run (errors are captured and diagnosed as "unresolved" but no fix is generated or applied). |
| `AI_FIXER_ENABLED` | Kill switch for the whole remediation pipeline. |
| `PROD_BRANCH` / `GIT_REMOTE` | Where validated fixes get pushed. |
| `TEST_COMMAND` | The automated gate a fix must pass before deploy. |
| `DEPLOY_WEBHOOK_URL` | Optional: called after a successful push, e.g. to kick a real deploy. |

## Run it

```bash
npm install
cp .env.example .env
npm start          # serves on :3000
npm test           # runs the test suite (also used as the AI fixer's gate)
```
