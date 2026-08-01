# parts-log-service

A universal activity log for software "parts" (services, modules, jobs — any
named unit of your system), with a deterministic error catcher and an
autonomous AI remediation loop sitting behind it that fixes, redeploys, and
live-validates before promoting to prod.

## How it fits together

```
external caller ──▶ POST /api/parts          (log an event for any part, addable at runtime)
external caller ──▶ POST /api/registry        (register/describe a part: its source file, aliases, metadata)
external caller ──▶ POST /api/errors          (report an error from outside this process)
in-process error ──▶ errorLogger middleware   (plain try/catch style capture - NO AI here)
                          │
                          ▼
                    services/remediationLoop.js   (the only place AI is invoked)
```

`remediationLoop.remediate()` repeats this cycle (up to `MAX_REMEDIATION_CYCLES`
times) for each captured error:

1. **Retrieve** recent log entries for the affected part.
2. **Analyze** — look up the part's source file in the registry.
3. **Fix** the root cause — ask Claude for a corrected file, apply it, and run
   `TEST_COMMAND` as a fast local gate (fail → revert, retry).
4. **Push** the fix to `DEV_BRANCH`.
5. **Restart** the Kubernetes deployment/pod for the part (`services/k8s.js`,
   skipped if `K8S_DEPLOYMENT` isn't set).
6. **Execute the affected APIs** against the redeployed service using a
   Postman collection (`services/postmanRunner.js`, skipped/treated as
   passing if `POSTMAN_COLLECTION_PATH` isn't set).
7. **Monitor** application logs for a window after the run
   (`services/logMonitor.js`).
8. If any API still fails or a new error shows up, the failure is fed back
   into the next AI request and the loop **repeats from step 1**.
9. Once everything passes, the same commit is **promoted to `PROD_BRANCH`** —
   no human step (`services/deploy.js`).
10. Monitoring **continues** for `POST_DEPLOY_MONITOR_WINDOW_MS` after
    promotion to confirm the service stays stable; anything that shows up
    after that is just a new error, captured and remediated the same way.

Every attempt of every cycle, plus the final outcome, is recorded on the
error entry (`GET /api/errors/:id` → `history`), so the whole loop is
auditable end to end.

```
                    .github/workflows/deploy-prod.yml  (fires on push to prod, re-runs tests, deploys)
```

Nothing here is hardcoded to a specific "part": what a part is, what file
backs it, what Kubernetes deployment backs it, and what aliases/key-commands
refer to it are all supplied through the API/config at runtime
(`POST /api/registry`). Add a new part by registering it - no code change
required. Every external tool in the loop (git, kubectl, Postman) degrades
to a no-op "skipped" step when unconfigured, so the whole pipeline runs
end-to-end with zero infrastructure and gets stricter as you wire in real
targets.

## Why the AI only sees captured errors, never raw requests

`errorLogger` is intentionally dumb: it's an Express error-handling
middleware that logs whatever exception reached it, with no branching logic
or model calls. AI only enters the picture afterwards, as a separate
consumer reading from the same error log (`services/remediationLoop.js`).
That keeps error *detection* deterministic and testable, and confines the
AI's blast radius to *remediation* of already-captured, already-logged
issues.

## Autonomy and safety rails

Per project requirements this pipeline is **fully autonomous**: a validated
fix is promoted to `PROD_BRANCH` with no human approval step. The gates are
all automated instead:

- the fix must pass `TEST_COMMAND` locally before it's even pushed to dev;
- once on dev, the Kubernetes restart must succeed (if configured);
- the Postman collection run against the redeployed service must fully pass
  (if configured);
- the log monitor must see nothing new for the part during/after that run;
- `MAX_REMEDIATION_CYCLES` bounds the retry loop so a persistently broken
  fix eventually gives up (`fix_failed`) instead of looping forever.

Review `src/services/remediationLoop.js` and `src/services/deploy.js` before
pointing `DEPLOY_WEBHOOK_URL` / `K8S_DEPLOYMENT` /
`.github/workflows/deploy-prod.yml` at real infrastructure - the default
behavior only pushes git branches, which is safe against any repo, but
wiring it to real infra raises the stakes of a bad AI fix considerably.
Consider branch protection / required status checks on `PROD_BRANCH` as a
second automated gate even though there's no human one.

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
Inspect the error/remediation audit trail, including each cycle's outcome
in `history`. `status` is one of `new`, `fixing`, `fixed`, `fix_failed`,
`unresolved`, `skipped`.

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables real AI remediation. Unset = dry-run (errors are captured and diagnosed as "unresolved" but no fix is generated or applied). |
| `AI_FIXER_ENABLED` | Kill switch for the whole remediation loop. |
| `DEV_BRANCH` / `PROD_BRANCH` / `GIT_REMOTE` | Where fixes are pushed and, once validated, promoted. |
| `TEST_COMMAND` | Fast local gate a fix must pass before it's even pushed to dev. |
| `MAX_REMEDIATION_CYCLES` | Retry bound for the retrieve/analyze/fix/deploy/test loop. |
| `K8S_NAMESPACE` / `K8S_DEPLOYMENT` / `KUBECTL_PATH` / `K8S_ROLLOUT_TIMEOUT` | Restarts the affected deployment after pushing to dev; skipped if `K8S_DEPLOYMENT` is blank. |
| `POSTMAN_COLLECTION_PATH` / `POSTMAN_BASE_URL` | Collection run against the redeployed service (see `postman/example.postman_collection.json`); skipped (treated as passing) if blank. |
| `MONITOR_WINDOW_MS` / `MONITOR_POLL_INTERVAL_MS` | How long/often to watch for new errors after each test cycle. |
| `POST_DEPLOY_MONITOR_WINDOW_MS` | How long to keep watching after a fix is promoted to prod. |
| `DEPLOY_WEBHOOK_URL` | Optional: called after every push (dev) and promotion (prod), e.g. to kick a real deploy. |

## Run it

```bash
npm install
cp .env.example .env
npm start          # serves on :3000
npm test           # runs the test suite (also used as the fast local gate)
```
