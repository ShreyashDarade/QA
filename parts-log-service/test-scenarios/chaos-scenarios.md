# Chaos Testing Scenarios — Autonomous Debugging & Deployment Agent

Test fixtures for validating `remediationLoop.js` (`parts-log-service/src/services/remediationLoop.js`)
against realistic production incidents. Each scenario is written to be fed to the agent (or a
human dry-running the agent's steps) as a starting `errors` entry / log dump, and describes what
a correct run through the 12-step loop should look like:

1. Read application logs → 2. Identify root cause → 3. Locate affected file → 4. Apply fix →
5. Run unit tests → 6. Commit → 7. Push to dev branch → 8. Restart K8s deployment/pod →
9. Execute affected Postman collections → 10. Monitor logs → 11. Repeat until all APIs pass →
12. Continue monitoring for new failures.

## How to use this file

Each scenario is self-contained. `Whether Code Changes Are Required` tells you if step 3-6 apply
at all — several incidents here are pure-infrastructure (`ImagePullBackOff`, disk full) where the
correct agent behavior is to **not** touch source code and instead fix config/K8s state directly.
`Expected Final Outcome` marked **multi-iteration** means the first fix attempt is expected to be
insufficient — the agent should detect the residual/new failure in step 10-11 and loop again with
that failure folded into the next analysis, per `remediationLoop.js`'s retry-with-context design.

## Coverage matrix

| # | ID | Category | Difficulty | Iterations |
|---|----|----------|------------|------------|
| 1 | CHAOS-01 | Python / Flask unhandled exception | Easy | Single |
| 2 | CHAOS-02 | Node.js / Express unhandled exception | Easy | Single |
| 3 | CHAOS-03 | FastAPI / API contract & JSON schema mismatch | Medium | Single |
| 4 | CHAOS-04 | Django / missing migration | Medium | Single |
| 5 | CHAOS-05 | Java Spring Boot / NPE + missing config property (multi-root) | Medium | **Multi** |
| 6 | CHAOS-06 | Python dependency conflict / CrashLoopBackOff | Hard | Single |
| 7 | CHAOS-07 | Missing environment variable / CrashLoopBackOff | Easy | Single |
| 8 | CHAOS-08 | Kubernetes CrashLoopBackOff — misconfigured liveness probe | Medium | **Multi** |
| 9 | CHAOS-09 | ImagePullBackOff — bad image tag | Easy | Single |
| 10 | CHAOS-10 | OOMKilled — Node.js memory leak | Hard | **Multi** |
| 11 | CHAOS-11 | Failed readiness probe — DB pool exhausted at boot | Medium | Single |
| 12 | CHAOS-12 | Network timeout — no timeout/retry on downstream call | Medium | Single |
| 13 | CHAOS-13 | PostgreSQL — max_connections exceeded + pool misconfig (multi-root) | Hard | **Multi** |
| 14 | CHAOS-14 | MySQL — deadlock on concurrent updates | Medium | Single |
| 15 | CHAOS-15 | MongoDB — duplicate key race condition | Medium | Single |
| 16 | CHAOS-16 | Redis — connection refused, no reconnect logic | Medium | Single |
| 17 | CHAOS-17 | Elasticsearch — red cluster + mapping mismatch (multi-root) | Hard | **Multi** |
| 18 | CHAOS-18 | Kafka — consumer rebalance storm + offset commit failure | Hard | **Multi** |
| 19 | CHAOS-19 | AWS — expired IAM credentials, S3 AccessDenied | Medium | Single |
| 20 | CHAOS-20 | JWT expiration / clock skew — cascading 401s | Medium | Single |
| 21 | CHAOS-21 | SSL certificate expired — TLS handshake failures (multi-root) | Hard | **Multi** |
| 22 | CHAOS-22 | Cascading multi-service failure: Redis outage → cache stampede → DB overload → auth 500s → checkout failures | Expert | **Multi** |

8 of 22 scenarios (36%) require multiple remediation cycles, exceeding the 30% target.

---

## CHAOS-01 — Python/Flask `NoneType` AttributeError

- **Difficulty:** Easy
- **Failure Category:** Python exception / Flask error
- **Initial Symptoms:** `GET /api/orders/{id}` returns HTTP 500 whenever an order has no assigned
  courier yet; error rate on this endpoint jumps to 100% after a promo drives a spike in new
  (uncourierable) orders.
- **Sample Error Logs:**
  ```
  [2026-08-01 09:14:02] ERROR in app: Exception on /api/orders/48213/status [GET]
  Traceback (most recent call last):
    File "/app/routes/orders.py", line 41, in get_order_status
      return jsonify({"courier_name": order.courier.name, "eta": order.courier.eta})
  AttributeError: 'NoneType' object has no attribute 'name'
  127.0.0.1 - - [01/Aug/2026 09:14:02] "GET /api/orders/48213/status HTTP/1.1" 500 -
  ```
- **Root Cause:** `order.courier` is `None` until a courier is assigned; `get_order_status` never
  null-checks before attribute access.
- **Expected Files to Modify:** `app/routes/orders.py`
- **Code Changes Required:** Yes — guard the `None` case and return a documented "unassigned"
  status instead of raising.
- **Kubernetes Action Required:** None beyond the standard rolling restart after deploy.
- **Database Changes Required:** None.
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** `GET /api/orders/{id}/status` (for orders with
  `courier_id IS NULL`).
- **Expected Final Outcome:** Single-iteration fix. Endpoint returns HTTP 200 with
  `{"courier_name": null, "eta": null, "status": "unassigned"}` for uncourierable orders.
- **Success Criteria:** Unit test covering `courier=None` passes; Postman run on the endpoint
  returns 200 for both assigned and unassigned orders; no new 500s in the 10-minute post-deploy
  log window.

---

## CHAOS-02 — Node.js/Express Uncaught TypeError

- **Difficulty:** Easy
- **Failure Category:** Node.js error
- **Initial Symptoms:** `POST /api/cart/items` intermittently 500s; correlates with clients on an
  older app version that omits the `quantity` field, defaulting to `undefined`.
- **Sample Error Logs:**
  ```
  TypeError: Cannot read properties of undefined (reading 'toFixed')
      at calculateLineTotal (/srv/app/src/cart/pricing.js:22:34)
      at addItem (/srv/app/src/cart/routes.js:58:19)
      at Layer.handle [as handle_request] (/srv/app/node_modules/express/lib/router/layer.js:95:5)
  {"level":"error","msg":"unhandled route error","route":"POST /api/cart/items","status":500,"errorId":"e5f1c2..."}
  ```
- **Root Cause:** `calculateLineTotal(price, quantity)` multiplies `price * quantity` without
  validating `quantity`; missing/undefined `quantity` propagates as `NaN` and then crashes on
  `.toFixed()`.
- **Expected Files to Modify:** `src/cart/pricing.js`
- **Code Changes Required:** Yes — validate/default `quantity` to `1` and reject negative or
  non-numeric values with a 400, not a 500.
- **Kubernetes Action Required:** Standard rolling restart.
- **Database Changes Required:** None.
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/cart/items` (payload without `quantity`).
- **Expected Final Outcome:** Single-iteration fix; malformed payloads now return 400 with a
  clear validation error instead of crashing the request.
- **Success Criteria:** Unit test for missing/negative/non-numeric `quantity`; Postman collection
  (valid payload, missing-quantity payload, negative-quantity payload) all return expected status
  codes; error log rate for this route returns to baseline.

---

## CHAOS-03 — FastAPI: API Contract Change / JSON Schema Mismatch

- **Difficulty:** Medium
- **Failure Category:** FastAPI error / API contract change
- **Initial Symptoms:** Downstream `billing-service` starts sending `amount` as a string
  (`"19.99"`) instead of a float after its own release; `payments-service`'s Pydantic model
  rejects every request with 422.
- **Sample Error Logs:**
  ```
  pydantic.error_wrappers.ValidationError: 1 validation error for ChargeRequest
  amount
    value is not a valid float (type=type_error.float)
  INFO:     10.2.3.14:0 - "POST /api/charges HTTP/1.1" 422 Unprocessable Entity
  {"event": "request_validation_failed", "path": "/api/charges", "errors": [{"loc": ["body", "amount"], "msg": "value is not a valid float"}]}
  ```
- **Root Cause:** Upstream contract drift — `billing-service` now serializes `amount` as a
  decimal-string to avoid float rounding, but `payments-service`'s `ChargeRequest` schema still
  declares `amount: float`.
- **Expected Files to Modify:** `app/schemas/charge.py` (accept `Decimal`/numeric strings via a
  validator), `app/services/charge_service.py` (use `Decimal` for money math instead of `float`).
- **Code Changes Required:** Yes.
- **Kubernetes Action Required:** Standard rolling restart.
- **Database Changes Required:** None (assuming the `amount` column is already `NUMERIC`; if it's
  `FLOAT`, a follow-up migration to `NUMERIC(12,2)` should be flagged, not silently skipped).
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/charges` with `amount` as a JSON string.
- **Expected Final Outcome:** Single-iteration fix; both string and numeric `amount` are accepted
  and processed with correct precision (no float rounding drift on charge totals).
- **Success Criteria:** Unit tests for string, float, and malformed `amount` inputs; Postman run
  against both the old (float) and new (string) payload shapes returns 200; a spot-check charge of
  `"19.99"` results in exactly `1999` cents stored, not `1998` or `2000`.

---

## CHAOS-04 — Django: Missing Migration

- **Difficulty:** Medium
- **Failure Category:** Django error / missing migration
- **Initial Symptoms:** New `loyalty_tier` field was added to the `Customer` model in code and
  deployed, but the migration wasn't generated/applied in prod; every customer-profile read fails.
- **Sample Error Logs:**
  ```
  django.db.utils.ProgrammingError: column customers_customer.loyalty_tier does not exist
  LINE 1: SELECT "customers_customer"."id", "customers_customer"."loy...
                                             ^
  File "/app/customers/views.py", line 33, in retrieve
    serializer = CustomerSerializer(customer)
  File "/app/customers/serializers.py", line 12, in to_representation
    return super().to_representation(instance)
  ```
- **Root Cause:** `0027_customer_loyalty_tier.py` migration file was never committed alongside the
  model change (developer ran `makemigrations` locally but didn't add the file to the commit).
- **Expected Files to Modify:** `customers/migrations/0027_customer_loyalty_tier.py` (generate and
  add), no application code changes needed since the model itself is already correct.
- **Code Changes Required:** Yes, but it's a generated migration file, not hand-written logic —
  the agent should run `manage.py makemigrations --check` to detect the drift and
  `makemigrations customers` to generate the missing file.
- **Kubernetes Action Required:** Run the migration as a pre-deploy Job/init-container, then
  restart the deployment.
- **Database Changes Required:** Yes — `ALTER TABLE customers_customer ADD COLUMN loyalty_tier
  varchar(20) NULL` (via the generated migration).
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** `GET /api/customers/{id}`, `GET /api/customers/`.
- **Expected Final Outcome:** Single-iteration fix once the migration is generated and applied.
- **Success Criteria:** `manage.py makemigrations --check` reports no drift; migration applies
  cleanly against a prod-like dataset; customer endpoints return 200 with `loyalty_tier` present
  (defaulting to `null` for existing rows).

---

## CHAOS-05 — Java Spring Boot: NPE + Missing Config Property (Multi-Root)

- **Difficulty:** Medium
- **Failure Category:** Java Spring Boot error / configuration
- **Initial Symptoms:** `notification-service` pod enters CrashLoopBackOff after a config-map
  update meant to add a new `notification.retry.max-attempts` property; pod logs show two
  distinct errors across restarts.
- **Sample Error Logs (attempt 1):**
  ```
  ***************************
  APPLICATION FAILED TO START
  ***************************

  Description:
  Binding to target org.springframework.boot.context.properties.bind.BindException: Failed to bind properties under 'notification.retry' to com.acme.notify.config.RetryProperties failed:

      Property: notification.retry.max-attempts
      Value: "five"
      Reason: failed to convert java.lang.String to int

  Action:
  Update your application's configuration
  ```
- **Sample Error Logs (attempt 2, after fixing the type but before fixing the NPE):**
  ```
  java.lang.NullPointerException: Cannot invoke "com.acme.notify.config.RetryProperties.getBackoffMs()" because "this.retryProperties" is null
      at com.acme.notify.service.RetryScheduler.schedule(RetryScheduler.java:29)
      at com.acme.notify.service.NotificationDispatcher.dispatch(NotificationDispatcher.java:47)
  ```
- **Root Cause:** Two independent problems introduced by the same config-map change: (1)
  `notification.retry.max-attempts=five` is a typo — should be a numeric value; (2)
  `RetryScheduler` isn't annotated `@Autowired`/constructor-injected for `RetryProperties`, so once
  the binding does succeed, the bean is still `null` at the injection point due to a missing
  `@Component` on `RetryProperties`.
- **Expected Files to Modify:** the deployment's `configmap.yaml` (fix the typo), and
  `RetryProperties.java` (add `@Component` / `@ConfigurationProperties(prefix = "notification.retry")`
  registration so it's an actual bean), `RetryScheduler.java` (constructor-inject it correctly).
- **Code Changes Required:** Yes, in addition to the config fix.
- **Kubernetes Action Required:** Update ConfigMap, restart deployment (twice — once per
  iteration below).
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes — `notification.retry.max-attempts=5`.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/notifications/send` (fails at
  dispatch/scheduling for any notification that requires retry logic; pod is unavailable for
  everything while CrashLoopBackOff persists).
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1 fixes the type-conversion error;
  the pod starts but the NPE surfaces on the first retry-eligible notification, which the log
  monitor step catches. Iteration 2 fixes the bean wiring; all notifications, including
  retry-triggering ones, succeed.
- **Success Criteria:** Pod reaches `Running`/`Ready` and stays up through two consecutive
  restarts; `POST /api/notifications/send` returns 200 including for payloads that force a retry
  path; no `NullPointerException` or `BindException` in logs for a 15-minute monitoring window.

---

## CHAOS-06 — Python Dependency Conflict / CrashLoopBackOff

- **Difficulty:** Hard
- **Failure Category:** Dependency conflict
- **Initial Symptoms:** After a routine `requirements.txt` bump (unrelated package upgraded),
  the `reporting-service` pod fails to start at all.
- **Sample Error Logs:**
  ```
  Traceback (most recent call last):
    File "/app/main.py", line 3, in <module>
      from reportlib import PdfBuilder
    File "/usr/local/lib/python3.11/site-packages/reportlib/__init__.py", line 5, in <module>
      from weasyprint import HTML
    File "/usr/local/lib/python3.11/site-packages/weasyprint/__init__.py", line 20, in <module>
      from pydyf import PDF
  ImportError: cannot import name 'PDF' from 'pydyf' (/usr/local/lib/python3.11/site-packages/pydyf/__init__.py)
  ```
  ```
  Kubernetes event: Back-off restarting failed container reporting-service in pod reporting-service-7d8f9c9c6b-2xk7q
  Warning  BackOff  12s (x8 over 3m)  kubelet  Back-off restarting failed container
  ```
- **Root Cause:** `requirements.txt` pinned `weasyprint==62.3` (transitively requires
  `pydyf>=0.10`) but a separate, unrelated pin of `pydyf==0.8.0` further down the file wins
  dependency resolution, leaving an incompatible `pydyf` installed. Classic transitive
  version-pin conflict introduced by an unrelated bump.
- **Expected Files to Modify:** `requirements.txt` (bump/unpin `pydyf`, or pin a
  `weasyprint`/`pydyf` pair known to be compatible), `requirements.lock`/`poetry.lock` equivalent
  if the repo uses one.
- **Code Changes Required:** Yes (dependency manifest, no application logic).
- **Kubernetes Action Required:** Rebuild the image with corrected dependencies, then restart the
  deployment (a plain pod restart without a rebuild will not fix this).
- **Database Changes Required:** None.
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** All endpoints — the pod never becomes ready, so
  every request to `reporting-service` times out or 503s at the ingress/gateway.
- **Expected Final Outcome:** Single-iteration fix (one dependency correction), but the deploy
  step is heavier than usual since it requires an image rebuild, not just a restart — the agent
  should recognize this is a Docker-build-affecting change and trigger CI to rebuild rather than
  just restarting the existing (broken) image.
- **Success Criteria:** `pip check` reports no conflicts; image builds cleanly in CI; pod reaches
  `Running`; `GET /health` and `POST /api/reports/generate` both return 200.

---

## CHAOS-07 — Missing Environment Variable / CrashLoopBackOff

- **Difficulty:** Easy
- **Failure Category:** Missing environment variable
- **Initial Symptoms:** `orders-service` pod restarts continuously right after a Helm values
  change removed what looked like an unused env var.
- **Sample Error Logs:**
  ```
  Error: Environment variable DATABASE_URL is not set
      at Object.<anonymous> (/srv/app/src/config.js:14:11)
      at Module._compile (node:internal/modules/cjs/loader:1256:14)
  Node.js v20.11.0 process exited with code 1
  ```
  ```
  Warning  BackOff    18s (x11 over 4m12s)  kubelet  Back-off restarting failed container orders-service in pod orders-service-6b6f7d8c9-p4mzt
  ```
- **Root Cause:** The Helm chart values change dropped the `DATABASE_URL` entry from the
  deployment's `envFrom` secretRef, believing it was dead config; it's actually read eagerly at
  module load time in `config.js`.
- **Expected Files to Modify:** Helm `values.yaml` / the deployment manifest (restore the
  `envFrom`/`secretKeyRef` for `DATABASE_URL`). No application code is at fault.
- **Code Changes Required:** No — this is a pure configuration regression.
- **Kubernetes Action Required:** Correct the Secret/env reference, then restart the deployment.
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes — restore `DATABASE_URL` in the deployment env config.
- **Postman APIs Expected to Fail (pre-fix):** All endpoints (pod never reaches Ready).
- **Expected Final Outcome:** Single-iteration fix.
- **Success Criteria:** Pod reaches `Running`/`Ready`; `GET /health` returns 200 with a DB
  connectivity check included; full Postman smoke collection passes.

---

## CHAOS-08 — Kubernetes CrashLoopBackOff: Misconfigured Liveness Probe

- **Difficulty:** Medium
- **Failure Category:** Kubernetes CrashLoopBackOff / probe misconfiguration
- **Initial Symptoms:** `search-service` was working fine until its JVM heap size was increased
  (to fix an unrelated OOM issue) — now it CrashLoopBackOffs even though the application itself
  starts successfully when run locally with the same heap settings.
- **Sample Error Logs / Events (attempt 1, before any fix):**
  ```
  Warning  Unhealthy  0s (x3 over 20s)  kubelet  Liveness probe failed: HTTP probe failed with statuscode: 000
  Warning  Killing    0s                kubelet  Container search-service failed liveness probe, will be restarted
  Normal   Pulled     0s                kubelet  Container image already present on machine
  ```
  Application log (proves the app *did* start, just slowly): `INFO  Started SearchApplication in 42.918 seconds (JVM running for 44.1)`
- **Root Cause:** `livenessProbe.initialDelaySeconds` is `15`, unchanged from before the heap
  increase; the larger heap now pushes JVM warm-up past 40s, so kubelet kills the container before
  it ever finishes starting — an infrastructure-timing bug that looks like an application crash
  but isn't one.
- **Expected Files to Modify:** `k8s/search-service/deployment.yaml`
  (`livenessProbe.initialDelaySeconds`, `readinessProbe.initialDelaySeconds`).
- **Code Changes Required:** No, this is purely a probe-timing config issue — **this is the trap
  in the scenario**: an agent that jumps straight to "fix the code" without reading the events
  closely will misdiagnose this as an application bug.
- **Kubernetes Action Required:** Increase `initialDelaySeconds` (and/or convert to a
  `startupProbe` with a generous `failureThreshold`), then restart the deployment.
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes (the probe timing values above).
- **Postman APIs Expected to Fail (pre-fix):** All endpoints — pod never stabilizes long enough
  to serve traffic.
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1: agent bumps
  `initialDelaySeconds` from 15 to 30, which is still insufficient under load (JVM warm-up
  degrades further because two pods are now flapping and competing for node CPU during the
  incident), and the pod still restarts, just less often. Iteration 2: agent switches to a
  `startupProbe` (`periodSeconds: 10`, `failureThreshold: 12` ⇒ 120s budget) instead of tuning
  `initialDelaySeconds` further, which fully resolves it regardless of node contention.
- **Success Criteria:** Pod reaches and stays `Ready` across 3 consecutive restarts under normal
  load; `GET /health` responds within the probe window every time; zero `Unhealthy`/`Killing`
  events over a 15-minute window.

---

## CHAOS-09 — ImagePullBackOff: Bad Image Tag

- **Difficulty:** Easy
- **Failure Category:** ImagePullBackOff
- **Initial Symptoms:** Deployment rollout for `pricing-service` never completes; old pods keep
  serving traffic (no outage), but the rollout is stuck and no new replicas come up.
- **Sample Error Logs / Events:**
  ```
  Failed to pull image "registry.acme.internal/pricing-service:v2.4.1": rpc error: code = NotFound
  desc = failed to pull and unpack image "registry.acme.internal/pricing-service:v2.4.1":
  failed to resolve reference: registry.acme.internal/pricing-service:v2.4.1: not found
  Warning  Failed     8s (x4 over 1m)  kubelet  Error: ImagePullBackOff
  ```
- **Root Cause:** The CI pipeline tagged and pushed the image as `v2.4.1-rc1` (a release-candidate
  suffix left in by a pipeline template change) but the deployment manifest was updated to request
  `v2.4.1` (without the suffix) — a tag typo/mismatch between CI output and the manifest.
- **Expected Files to Modify:** Deployment manifest / Helm values (`image.tag`), not the
  application source.
- **Code Changes Required:** No.
- **Kubernetes Action Required:** Correct the image tag reference and re-trigger the rollout
  (`kubectl rollout restart` after the manifest fix).
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes — the image tag in the deployment spec.
- **Postman APIs Expected to Fail (pre-fix):** None immediately visible (old replicas still
  serving) — but this should still be flagged as a failed deployment, since a genuine `v2.4.1`
  fix never actually ships.
- **Expected Final Outcome:** Single-iteration fix.
- **Success Criteria:** `kubectl rollout status` reports the new ReplicaSet fully available;
  `GET /health` on new pods reports the correct build/version; no `ImagePullBackOff` events for
  the corrected tag.

---

## CHAOS-10 — OOMKilled: Node.js Memory Leak (Event Listener Leak)

- **Difficulty:** Hard
- **Failure Category:** OOMKilled / memory leak
- **Initial Symptoms:** `websocket-gateway` pods restart every 40-90 minutes under sustained
  traffic; memory usage climbs steadily and linearly with connection churn, never plateaus.
- **Sample Error Logs / Events:**
  ```
  Warning  OOMKilling  0s   kernel  Memory cgroup out of memory: Killed process 1 (node) total-vm:2185432kB, anon-rss:1998112kB
  State:          Terminated
    Reason:       OOMKilled
    Exit Code:    137
  ```
  Metrics snippet: `container_memory_working_set_bytes{pod="websocket-gateway-6c9d..."}` rising
  from 180MB to 2GB over 70 minutes with a stable, cyclical connection count (no traffic spike).
- **Root Cause:** `ConnectionManager.onClient(socket)` registers a `socket.on('message', handler)`
  listener but the matching `removeListener` on disconnect was dropped in a refactor months ago;
  every reconnect adds a listener that's never cleaned up, and each retained listener closure
  holds a reference to the (now-dead) socket and its buffered message history.
- **Expected Files to Modify:** `src/gateway/connectionManager.js`
- **Code Changes Required:** Yes.
- **Kubernetes Action Required:** Restart deployment after each fix attempt; consider a temporary
  `memory` limit bump only as a stopgap, not a fix (should not be the agent's final action).
- **Database Changes Required:** None.
- **Configuration Changes Required:** None (aside from the temporary/optional limit bump above).
- **Postman APIs Expected to Fail (pre-fix):** Not a request-response API — surfaces as
  connection drops/`GET /health` returning 200 right after restart, then the pod becoming
  unreachable again after ~1hr, which is why this needs sustained log/metric monitoring rather
  than a single Postman pass to catch.
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1: agent adds
  `socket.removeAllListeners('message')` in the disconnect handler — this reduces the leak rate
  significantly (good enough to pass an immediate Postman smoke pass) but doesn't eliminate it,
  because `ConnectionManager` also keeps a `Map<socketId, socket>` entry that's never deleted on
  disconnect, so memory still grows, just slower. The extended monitoring window (step 12) catches
  the still-climbing memory trend that a quick pass would miss. Iteration 2: agent additionally
  deletes the `Map` entry in the disconnect handler, fully flattening memory usage.
- **Success Criteria:** `container_memory_working_set_bytes` stays flat (±10%) over a 2-hour
  soak window at steady connection churn; zero `OOMKilled` events in that window; connection
  count and listener count converge (no unbounded growth) as verified by a `/debug/listeners`
  introspection endpoint.

---

## CHAOS-11 — Failed Readiness Probe: DB Connection Pool Exhausted at Boot

- **Difficulty:** Medium
- **Failure Category:** Failed readiness probe / database connection failure
- **Initial Symptoms:** After scaling `checkout-service` from 4 to 12 replicas for a flash sale,
  roughly half the new pods never become Ready.
- **Sample Error Logs:**
  ```
  FATAL: remaining connection slots are reserved for non-replication superuser connections
  	at Connection.parseE (/srv/app/node_modules/pg/lib/connection.js:614:13)
  {"level":"error","msg":"readiness check failed: db ping timeout","route":"/ready"}
  ```
  ```
  Warning  Unhealthy  5s (x6 over 1m)  kubelet  Readiness probe failed: HTTP probe failed with statuscode: 503
  ```
- **Root Cause:** Each pod opens a fixed pool of 20 DB connections at boot;
  `12 replicas × 20 = 240` exceeds PostgreSQL's `max_connections = 200`, so pods that start last
  can't acquire enough connections to pass the readiness check.
- **Expected Files to Modify:** `src/db/pool.js` (reduce per-pod pool size and/or make it
  configurable via env so it can be tuned without a code change next time), plus the PgBouncer /
  RDS parameter group if one exists (`configmap`/infra-as-code file for `max_connections`).
- **Code Changes Required:** Yes (make pool size env-configurable and lower the default), in
  addition to the infra-side connection-limit review.
- **Kubernetes Action Required:** Restart deployment after the pool-size fix.
- **Database Changes Required:** Potentially — raising `max_connections` (or, preferably, fronting
  the DB with PgBouncer) if per-pod pool size alone can't be reduced enough for the target replica
  count.
- **Configuration Changes Required:** Yes — `DB_POOL_SIZE` env var added/tuned per replica count.
- **Postman APIs Expected to Fail (pre-fix):** All `checkout-service` endpoints intermittently,
  correlated with which pods failed readiness.
- **Expected Final Outcome:** Single-iteration fix (pool size reduction is sufficient at the
  target scale in this scenario).
- **Success Criteria:** All 12 replicas reach `Ready`; `SELECT count(*) FROM pg_stat_activity`
  stays comfortably under `max_connections` at full scale; checkout Postman collection passes at
  full replica count under simulated load.

---

## CHAOS-12 — Network Timeout: No Timeout/Retry on Downstream Call

- **Difficulty:** Medium
- **Failure Category:** Network timeout
- **Initial Symptoms:** `checkout-service` requests to `tax-calculation-service` occasionally hang
  for 30+ seconds and eventually the whole checkout request times out at the gateway, even though
  `tax-calculation-service` itself is healthy and just briefly slow (p99 latency spike, not down).
- **Sample Error Logs:**
  ```
  {"level":"error","msg":"gateway timeout","route":"POST /api/checkout","duration_ms":30012}
  UpstreamRequestTimeout: request to http://tax-calculation-service.internal/calculate exceeded 30000ms
      at ClientRequest.<anonymous> (/srv/app/src/clients/taxClient.js:18:9)
  ```
- **Root Cause:** `taxClient.js` uses the default `http.request` with no explicit timeout, so a
  single slow downstream call blocks the entire request indefinitely up to the gateway's outer
  30s timeout, instead of failing fast with a sensible fallback (e.g., a cached/estimated tax
  rate) within a few seconds.
- **Expected Files to Modify:** `src/clients/taxClient.js`
- **Code Changes Required:** Yes — add an explicit client-side timeout (e.g. 3s) with one retry
  and a fallback to a cached estimate on exhaustion, rather than propagating the hang.
- **Kubernetes Action Required:** Standard rolling restart.
- **Database Changes Required:** None.
- **Configuration Changes Required:** New `TAX_CLIENT_TIMEOUT_MS` env var (defaulted sensibly).
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/checkout` (fails intermittently,
  correlated with `tax-calculation-service` p99 latency).
- **Expected Final Outcome:** Single-iteration fix.
- **Success Criteria:** Checkout requests complete within 5s even when
  `tax-calculation-service` is artificially delayed 10s in a test; fallback tax estimate is
  clearly flagged in the response for reconciliation; no gateway-level timeouts in the monitoring
  window.

---

## CHAOS-13 — PostgreSQL: `max_connections` Exceeded + Pool Misconfiguration (Multi-Root)

- **Difficulty:** Hard
- **Failure Category:** PostgreSQL / database connection failures
- **Initial Symptoms:** Intermittent, seemingly random 500s across several services sharing the
  same RDS instance, worsening throughout the day and clearing overnight.
- **Sample Error Logs (attempt 1 diagnosis — looks single-cause):**
  ```
  psycopg2.OperationalError: FATAL:  sorry, too many clients already
  ```
- **Sample Error Logs (after the obvious pool-size fix, revealing the second root cause):**
  ```
  psycopg2.OperationalError: FATAL:  sorry, too many clients already
  # connection count still climbing even after all services' pool sizes were reduced and confirmed correct
  ```
  `SELECT state, count(*) FROM pg_stat_activity GROUP BY state;` shows hundreds of rows in
  `idle in transaction` state, held for 20+ minutes each.
- **Root Cause:** Two independent contributors: (1) too many services independently sized their
  pools without a shared connection budget against `max_connections=200` (the surface-level
  cause, fixed in iteration 1); (2) `inventory-service`'s `reserve_stock()` opens a transaction,
  makes an outbound HTTP call to a fraud-check service *while the transaction is still open*, and
  never sets a statement/idle-in-transaction timeout — under fraud-service latency, these
  connections pile up in `idle in transaction` and are never released, which iteration 1's pool
  resizing doesn't touch at all.
- **Expected Files to Modify:** Each service's pool-size config (iteration 1); 
  `inventory/services/reservation.py` to move the HTTP call outside the transaction (iteration 2);
  RDS parameter group for `idle_in_transaction_session_timeout` as a backstop.
- **Code Changes Required:** Yes, in both iterations.
- **Kubernetes Action Required:** Restart each affected deployment after its respective fix.
- **Database Changes Required:** Yes — set `idle_in_transaction_session_timeout = '30s'` at the
  database/role level so this class of bug fails fast in the future instead of exhausting
  connections.
- **Configuration Changes Required:** Yes — per-service `DB_POOL_SIZE` budget documented and
  enforced so the sum across all services stays under `max_connections` with headroom.
- **Postman APIs Expected to Fail (pre-fix):** Endpoints across multiple services
  (`checkout-service`, `inventory-service`, `orders-service`) intermittently, not just one —
  a strong signal this is a shared-resource incident, not a single-service bug.
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1 (pool-size rebalancing) reduces
  but does not eliminate the errors — `pg_stat_activity` still shows growing
  `idle in transaction` sessions from `inventory-service` specifically, which the log-monitoring
  step should surface as the next target. Iteration 2 (moving the HTTP call out of the
  transaction + statement timeout) resolves it fully.
- **Success Criteria:** `pg_stat_activity` connection count stays well under `max_connections`
  under peak simulated load for 30 minutes; zero `idle in transaction` sessions older than the
  configured timeout; all three affected services' Postman collections pass.

---

## CHAOS-14 — MySQL: Deadlock on Concurrent Updates

- **Difficulty:** Medium
- **Failure Category:** MySQL / deadlock
- **Initial Symptoms:** `POST /api/inventory/adjust` fails ~2% of the time during high-concurrency
  restocking windows with a generic 500.
- **Sample Error Logs:**
  ```
  pymysql.err.OperationalError: (1213, 'Deadlock found when trying to get lock; try restarting transaction')
    File "/app/inventory/adjust.py", line 31, in adjust_stock
      cursor.execute(UPDATE_WAREHOUSE_ROW, (delta, warehouse_id))
      cursor.execute(UPDATE_SKU_ROW, (delta, sku_id))
  ```
- **Root Cause:** `adjust_stock()` updates the `warehouse` row then the `sku` row, but a
  concurrent call to `transfer_stock()` updates the same two tables in the opposite order
  (`sku` then `warehouse`) — classic lock-ordering deadlock between two code paths.
- **Expected Files to Modify:** `inventory/adjust.py`, `inventory/transfer.py` (enforce a single,
  consistent lock acquisition order — e.g. always `warehouse` before `sku` — across both
  functions), plus adding bounded retry-with-backoff around deadlock errors specifically (MySQL
  error 1213) as a defense-in-depth measure.
- **Code Changes Required:** Yes.
- **Kubernetes Action Required:** Standard rolling restart.
- **Database Changes Required:** None (this is a query/transaction-ordering fix, not a schema
  change).
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/inventory/adjust`,
  `POST /api/inventory/transfer` under concurrent load (a single sequential Postman run may not
  reproduce this — the collection should include a concurrency/load-test variant, not just
  single-request checks).
- **Expected Final Outcome:** Single-iteration fix.
- **Success Criteria:** A concurrency test firing 50 concurrent `adjust`/`transfer` calls against
  overlapping SKUs/warehouses completes with zero unhandled deadlock errors (retried transactions
  are fine, surfaced 500s are not); row totals reconcile correctly afterward (no lost updates).

---

## CHAOS-15 — MongoDB: Duplicate Key Race Condition

- **Difficulty:** Medium
- **Failure Category:** MongoDB / duplicate records
- **Initial Symptoms:** Occasional 500s on `POST /api/users/signup` during traffic spikes;
  affected users report "email already in use" errors on their very first signup attempt.
- **Sample Error Logs:**
  ```
  pymongo.errors.DuplicateKeyError: E11000 duplicate key error collection: appdb.users index: email_1 dup key: { email: "j.rivera@example.com" }
    File "/app/users/signup.py", line 27, in create_user
      existing = users_collection.find_one({"email": email})
      if not existing:
          users_collection.insert_one({...})
  ```
- **Root Cause:** Classic check-then-act race: `find_one` then `insert_one` isn't atomic, so two
  concurrent signup requests with the same email (e.g. a double-submit from a slow client) both
  pass the `find_one` check before either inserts, and the second `insert_one` throws instead of
  being handled as "user already exists."
- **Expected Files to Modify:** `users/signup.py`
- **Code Changes Required:** Yes — rely on the unique index as the source of truth: attempt the
  insert directly and catch `DuplicateKeyError` to return a proper 409 Conflict, removing the
  redundant (and racy) pre-check.
- **Kubernetes Action Required:** Standard rolling restart.
- **Database Changes Required:** None (the `email_1` unique index already exists and is doing its
  job correctly — the bug is entirely in how the app handles the resulting exception).
- **Configuration Changes Required:** None.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/users/signup` under concurrent duplicate
  submissions (needs a concurrency-variant test, same caveat as CHAOS-14).
- **Expected Final Outcome:** Single-iteration fix.
- **Success Criteria:** Firing 20 concurrent signups with the same email results in exactly one
  `201 Created` and nineteen `409 Conflict` responses, zero `500`s; existing unique, non-colliding
  signups continue to succeed normally.

---

## CHAOS-16 — Redis: Connection Refused, No Reconnect Logic

- **Difficulty:** Medium
- **Failure Category:** Redis failure
- **Initial Symptoms:** After a routine Redis maintenance restart (expected, brief), the
  `session-service` never recovers even though Redis itself came back up within 15 seconds.
- **Sample Error Logs:**
  ```
  Error: connect ECONNREFUSED 10.4.2.11:6379
      at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1595:16)
  {"level":"error","msg":"redis client error","event":"error"}
  # repeats once, then nothing further logged - client gave up silently
  ```
- **Root Cause:** The Redis client was instantiated with `retryStrategy: () => null` (retries
  disabled, likely copy-pasted from an example that intentionally disabled it for a script), so
  the very first connection drop kills the client permanently with no reconnect attempt for the
  life of the process.
- **Expected Files to Modify:** `src/cache/redisClient.js`
- **Code Changes Required:** Yes — configure exponential-backoff reconnection
  (`retryStrategy: (times) => Math.min(times * 100, 5000)`) and add a `/health` check that reports
  `503` while Redis is disconnected so Kubernetes can correctly reflect degraded state instead of
  reporting healthy while every session lookup silently fails.
- **Kubernetes Action Required:** Standard rolling restart after the fix; no restart would have
  been needed at all if reconnect logic had existed in the first place.
- **Database Changes Required:** None.
- **Configuration Changes Required:** None (reconnect strategy is code, not config, in this
  client).
- **Postman APIs Expected to Fail (pre-fix):** Any endpoint requiring session lookup
  (`GET /api/me`, `POST /api/cart/*`) — all fail with 401/500 indefinitely after the Redis blip,
  long after Redis itself is healthy again.
- **Expected Final Outcome:** Single-iteration fix.
- **Success Criteria:** Simulated Redis restart (kill + restart the Redis pod) results in the
  client reconnecting automatically within the backoff window with no service restart required;
  `/health` correctly reports `503` during the outage window and `200` after reconnection;
  session-dependent endpoints resume succeeding automatically once reconnected.

---

## CHAOS-17 — Elasticsearch: Red Cluster Health + Mapping Mismatch (Multi-Root)

- **Difficulty:** Hard
- **Failure Category:** Elasticsearch failure
- **Initial Symptoms:** `search-service`'s `/api/search` endpoint starts returning 503s
  right after a new field (`price_range`) was added to product documents; simultaneously,
  cluster health degrades from green to red.
- **Sample Error Logs (attempt 1):**
  ```
  elasticsearch.exceptions.RequestError: RequestError(400, 'illegal_argument_exception',
  'mapper [price_range] of different type, current_type [keyword], merged_type [long]')
  ```
  ```
  GET _cluster/health
  {"status": "red", "unassigned_shards": 3, "active_shards_percent_as_number": 91.6}
  ```
- **Root Cause:** Two independent issues surfaced by the same deploy: (1) `price_range` was
  indexed as a `keyword` on day one by dynamic mapping (someone's test data was string-typed) and
  the new code now sends it as a `long`, which Elasticsearch's dynamic mapping refuses to merge —
  an app-level type mismatch; (2) unrelated to the mapping issue, one data node ran out of disk
  above the `cluster.routing.allocation.disk.watermark.high` threshold the same day, which is why
  cluster health is red with unassigned shards, not just yellow — fixing the mapping alone would
  not restore full cluster health.
- **Expected Files to Modify:** `src/search/productIndexer.js` (send `price_range` consistently
  as a `long`, or better, add an explicit mapping template instead of relying on dynamic mapping),
  plus a reindex of the `products` index with the corrected explicit mapping; infra-side, free up
  or expand disk on the affected data node.
- **Code Changes Required:** Yes (indexer + explicit mapping template).
- **Kubernetes Action Required:** Restart `search-service` after the indexer fix; separately,
  scale/expand the Elasticsearch data node's PVC (or add a node) to clear the disk watermark —
  this is infra, not app-pod, remediation.
- **Database Changes Required:** N/A (Elasticsearch, not a relational DB) — but functionally
  equivalent: an index remapping/reindex is required, which is a "schema change" for this
  datastore.
- **Configuration Changes Required:** Yes — explicit index mapping template for `products`.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/products` (indexing fails on 400),
  `GET /api/search` (503 while cluster is red).
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1 fixes the mapping/indexer code;
  indexing succeeds again, but `GET /api/search` still intermittently 503s because the cluster is
  still red from the disk-watermark issue, which the log/cluster-health monitoring step should
  catch. Iteration 2 addresses the disk watermark (infra action, not a code change) and confirms
  cluster health returns to green.
- **Success Criteria:** `_cluster/health` reports `green` with `active_shards_percent_as_number:
  100`; `POST /api/products` with a `price_range` value succeeds; `GET /api/search` returns
  correct results filtering on `price_range` as a numeric range query.

---

## CHAOS-18 — Kafka: Consumer Rebalance Storm + Offset Commit Failure (Multi-Root)

- **Difficulty:** Hard
- **Failure Category:** Kafka failure
- **Initial Symptoms:** `order-events` consumer group for `fulfillment-service` shows
  ever-increasing lag; downstream fulfillment updates are delayed by 20+ minutes and climbing.
- **Sample Error Logs (attempt 1):**
  ```
  [Consumer clientId=fulfillment-consumer-3, groupId=fulfillment-service] Attempt to heartbeat failed since group is rebalancing
  [Consumer clientId=fulfillment-consumer-3, groupId=fulfillment-service] Member fulfillment-consumer-3-e4a1... sending LeaveGroup request due to consumer poll timeout has expired
  ```
- **Sample Error Logs (after the poll-timeout fix, revealing the second root cause):**
  ```
  org.apache.kafka.clients.consumer.CommitFailedException: Commit cannot be completed since the group has already rebalanced and assigned the partitions to another member.
      at FulfillmentConsumer.handleBatch(FulfillmentConsumer.java:88)
  # duplicate order-fulfillment side effects observed for ~40 orders during the rebalance window
  ```
- **Root Cause:** Two compounding issues: (1) `max.poll.interval.ms` is left at the default 5
  minutes, but a recent change made `handleBatch()` call a slow synchronous inventory-check API
  per message, so large batches routinely exceed the poll interval and trigger a rebalance
  (iteration-1 fix: batch-level async processing + raise `max.poll.interval.ms` appropriately, not
  just raise the timeout blindly); (2) `handleBatch()` commits offsets *after* triggering
  side-effecting fulfillment actions rather than processing idempotently, so the rebalances caused
  by issue (1) result in re-delivered messages producing duplicate fulfillment actions — a
  correctness bug that issue (1)'s fix alone doesn't address.
- **Expected Files to Modify:** `FulfillmentConsumer.java` (both the batch-processing/poll-timing
  fix and, separately, making `handleBatch` idempotent via an `order_id` dedupe check before
  triggering fulfillment), consumer config (`max.poll.interval.ms`, `max.poll.records`).
- **Code Changes Required:** Yes, in both iterations.
- **Kubernetes Action Required:** Restart the consumer deployment after each fix.
- **Database Changes Required:** Yes — add a `fulfillment_dedupe(order_id, processed_at)` table
  (or equivalent) backing the idempotency check added in iteration 2.
- **Configuration Changes Required:** Yes — tuned `max.poll.interval.ms` / `max.poll.records`.
- **Postman APIs Expected to Fail (pre-fix):** Not a direct HTTP API failure — surfaces as lag on
  `GET /api/fulfillment/queue-depth` (an internal metrics endpoint) exceeding SLA, and as
  duplicate-fulfillment records visible via `GET /api/orders/{id}/fulfillment-history`.
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1 resolves the rebalance storm and
  lag; monitoring then surfaces the duplicate-fulfillment side effect as a new, distinct failure
  (visible in `fulfillment-history` audit records, not consumer logs) that iteration 2 fixes via
  idempotency.
- **Success Criteria:** Consumer lag stays near zero under normal throughput; zero
  `CommitFailedException`s during a simulated slow-batch load test; each `order_id` has exactly
  one fulfillment record even when a rebalance is deliberately forced mid-batch during testing.

---

## CHAOS-19 — AWS: Expired IAM Credentials, S3 AccessDenied

- **Difficulty:** Medium
- **Failure Category:** AWS credential failure / S3 access failure
- **Initial Symptoms:** `POST /api/documents/upload` starts failing for all users at the same
  moment; nothing was deployed around that time.
- **Sample Error Logs:**
  ```
  botocore.exceptions.ClientError: An error occurred (ExpiredToken) when calling the PutObject operation:
  The provided token has expired.
  ```
  then, after a naive credential refresh attempt with the wrong policy attached:
  ```
  botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the PutObject operation:
  User: arn:aws:sts::123456789012:assumed-role/documents-service-role/i-0a1b2c3d is not authorized
  to perform: s3:PutObject on resource: "arn:aws:s3:::acme-docs-prod/*" because no identity-based
  policy allows the s3:PutObject action
  ```
- **Root Cause:** The pod's IRSA (IAM Roles for Service Accounts) annotation was pointing at a
  role whose trust policy had a `session-duration` that just expired and wasn't set to
  auto-renew; the on-call's first response (attaching a quick, overly-narrow inline policy while
  investigating) fixed authentication but not authorization, since the replacement policy didn't
  include `s3:PutObject`, only `s3:GetObject`.
- **Expected Files to Modify:** IAM role/trust-policy definition (Terraform/CloudFormation, not
  application code) — fix the session-duration/trust policy, and correct the attached policy to
  include the required S3 actions matching least-privilege for this service.
- **Code Changes Required:** No application code changes required — this is entirely an
  infra/IAM-config incident, and the agent should recognize that and not go looking for a bug in
  `documents/upload.py`.
- **Kubernetes Action Required:** Restart the deployment once the IRSA annotation/role is fixed so
  pods pick up fresh credentials (or wait for automatic token refresh, if configured correctly
  going forward).
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes — IAM trust policy and permissions policy.
- **Postman APIs Expected to Fail (pre-fix):** `POST /api/documents/upload`,
  `GET /api/documents/{id}/download` (if download also touches S3 and shares the role).
- **Expected Final Outcome:** Single-iteration fix once the correct policy (not the narrow
  stopgap) is applied — treated as one scenario since the "wrong policy" step is presented here as
  the realistic first response an on-call would try, folded into the root-cause analysis rather
  than as a separate agent iteration.
- **Success Criteria:** `aws sts get-caller-identity` from within the pod resolves cleanly;
  `PutObject` and `GetObject` both succeed against the target bucket/prefix; no `ExpiredToken` or
  `AccessDenied` errors over a 30-minute window including a forced credential-refresh cycle.

---

## CHAOS-20 — JWT Expiration / Clock Skew — Cascading 401s

- **Difficulty:** Medium
- **Failure Category:** Authentication failure / JWT expiration
- **Initial Symptoms:** After a new node pool was added to the cluster, a subset of requests
  (roughly matching the new nodes) starts failing auth even with freshly issued tokens.
- **Sample Error Logs:**
  ```
  {"level":"warn","msg":"jwt verification failed","error":"jwt expired","exp":1785600000,"now":1785600041}
  # token's exp is only 41 seconds in the past despite being issued 2 minutes ago with a 5-minute TTL
  ```
  `chronyc tracking` on an affected node: `System time     : 47.812311 seconds fast of NTP time`
- **Root Cause:** New nodes were provisioned from an AMI with `chronyd` disabled by a hardening
  script that (incorrectly) treated NTP as an unnecessary service; their clocks drift ~48 seconds
  fast within hours, which is enough for `auth-service`'s JWT validation (`exp` check with zero
  leeway) to reject tokens that are actually still valid.
- **Expected Files to Modify:** Node bootstrap/hardening script (re-enable `chronyd`), and,
  defensively, `src/auth/verifyToken.js` to add a small `clockTolerance` (e.g. 10s) to JWT
  validation so minor drift doesn't cause hard failures while NTP is the real fix.
- **Code Changes Required:** Yes, for the defensive `clockTolerance` addition; the primary fix is
  infra (NTP), not application code.
- **Kubernetes Action Required:** No pod restart fixes clock drift — this requires fixing node
  time sync (out-of-band, via the node bootstrap config) and, once nodes are correct,
  cordoning/replacing the already-drifted nodes rather than waiting for chrony to slowly correct a
  48s offset.
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes — node bootstrap hardening script, `clockTolerance` in
  the JWT verification config.
- **Postman APIs Expected to Fail (pre-fix):** Any authenticated endpoint, intermittently,
  correlated with which node handled the request — a strong "why does this only fail sometimes"
  signal pointing at node-level rather than app-level state.
- **Expected Final Outcome:** Single-iteration fix (from the app's perspective — the
  `clockTolerance` change plus draining the affected nodes resolves it without needing a second
  code iteration).
- **Success Criteria:** All nodes report `chronyc tracking` offset under 1 second; authenticated
  requests succeed uniformly across all nodes; JWT rejection rate returns to its (near-zero)
  baseline.

---

## CHAOS-21 — SSL Certificate Expired — TLS Handshake Failures (Multi-Root)

- **Difficulty:** Hard
- **Failure Category:** SSL certificate issue
- **Initial Symptoms:** External partner integrations calling `partner-api` over mTLS start
  failing simultaneously at midnight; internal service-to-service calls are unaffected.
- **Sample Error Logs (attempt 1 — renew the cert):**
  ```
  SSL routines:ssl3_read_bytes:sslv3 alert certificate expired
  curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL in connection to partner-api.acme.com:443
  ```
- **Sample Error Logs (after renewal, revealing the second root cause):**
  ```
  x509: certificate signed by unknown authority
  # partners' clients pin the old intermediate CA and reject the new cert chain
  ```
- **Root Cause:** Two issues: (1) the leaf certificate for `partner-api.acme.com` expired because
  the cert-manager `Certificate` resource's renewal job had been silently failing for weeks due to
  a DNS-01 challenge misconfiguration after a DNS provider migration (fixed in iteration 1 by
  correcting the DNS-01 solver config and forcing a renewal); (2) the renewal was issued from a
  *different* intermediate CA than the original (the CA provider rotated intermediates between
  issuances), and several partners hard-pin the old intermediate in their trust store, so their
  clients reject the otherwise-valid new cert — a downstream compatibility issue only discoverable
  once the primary renewal succeeds.
- **Expected Files to Modify:** cert-manager `Certificate`/`Issuer` resource (DNS-01 solver
  config) for iteration 1; for iteration 2, the ingress/gateway TLS config to serve the *full
  chain* including the previous intermediate during the transition window (chain-of-trust overlap)
  rather than just the leaf + new intermediate.
- **Code Changes Required:** No application code changes — entirely a TLS/cert-manager/ingress
  configuration incident.
- **Kubernetes Action Required:** Restart cert-manager's challenge solver pod after config fix;
  restart/reload the ingress controller to pick up the renewed cert and updated chain.
- **Database Changes Required:** None.
- **Configuration Changes Required:** Yes — DNS-01 solver config, ingress TLS chain configuration.
- **Postman APIs Expected to Fail (pre-fix):** All `partner-api` endpoints for external/mTLS
  callers specifically (internal calls over the service mesh's own mTLS are unaffected, which is
  itself a diagnostic clue pointing at the external-facing cert, not app logic).
- **Expected Final Outcome:** **Multi-iteration.** Iteration 1 (cert renewal) fixes the expiry but
  a subset of partners still fail post-renewal with a *different* error, caught by the log
  monitoring step distinguishing "certificate expired" from "unknown authority" failures.
  Iteration 2 (serving the overlapping chain) resolves it for all partners.
- **Success Criteria:** `openssl s_client -connect partner-api.acme.com:443` shows a
  non-expired cert with a chain all affected partners' pinned trust stores accept; zero TLS
  handshake failures across a full partner smoke-test list for 24 hours; cert-manager renewal job
  succeeds automatically ahead of the next expiry with margin to spare.

---

## CHAOS-22 — Cascading Multi-Service Failure: Redis Outage → Cache Stampede → DB Overload → Auth 500s → Checkout Failures

- **Difficulty:** Expert
- **Failure Category:** Cascading failure across microservices (Redis failure + database
  connection failure + authentication failure + downstream API failures, compounded)
- **Initial Symptoms:** A brief, planned Redis failover (30s) triggers a full-site outage lasting
  25+ minutes, wildly disproportionate to the triggering event.
- **Sample Error Logs / Timeline:**
  ```
  T+0s   Redis primary failover begins (planned, expected ~30s blip)
  T+2s   {"svc":"session-cache","level":"warn","msg":"redis unavailable, falling back to DB lookup"}
  T+5s   # every service configured to fall back to Postgres on cache miss does so SIMULTANEOUSLY -
         # a classic cache stampede, not accounted for in any service's fallback design
  T+8s   postgres: FATAL: sorry, too many clients already (auth-service, session lookups)
  T+9s   {"svc":"auth-service","level":"error","msg":"session validation failed","error":"db timeout"} x 4,800/min
  T+11s  {"svc":"api-gateway","level":"error","msg":"upstream auth-service 500","route":"*"}
  T+14s  {"svc":"checkout-service","level":"error","msg":"401 from auth-service, treating as unauthenticated","route":"POST /api/checkout"}
  T+30s  Redis failover completes, new primary healthy
  T+31s  # DB connections still saturated with backlog from T+5-30s; auth-service keeps timing out
  T+180s auth-service pool finally drains backlog; error rate starts declining
  T+25min system fully recovered as retry storms from client-side auto-retry logic finally taper off
  ```
- **Root Cause (multi-root, three independent contributing designs, none of which is a single
  "bug" but which compound into an outage):**
  1. Every service's Redis-fallback path queries Postgres directly with no synchronization
     (no single-flight/lock, no jitter, no circuit breaker) — a 30s Redis blip becomes a
     thundering herd against the database.
  2. `auth-service`'s DB pool has no dedicated headroom separate from other query types, so
     session-lookup fallback traffic competes for the same limited connections as everything
     else, and once exhausted, legitimate session validations fail even for users who were
     already authenticated before the incident (their sessions were cached, now they aren't).
  3. Client-side retry logic in the mobile/web apps has no backoff or jitter, so once
     `checkout-service` starts returning errors, retried requests amplify the load further,
     extending recovery well past when Redis and the DB pool would have naturally recovered on
     their own.
- **Expected Files to Modify:** `session-cache` fallback logic (add single-flight de-duplication
  + jitter + a circuit breaker that fails fast to a cached "assume valid, revalidate soon" mode
  instead of every request hitting Postgres) across each affected service; `auth-service` DB pool
  config (dedicated headroom / a separate read-replica for fallback lookups); client SDKs
  (retry-with-exponential-backoff-and-jitter, respecting `Retry-After`).
- **Code Changes Required:** Yes, across `session-cache` client library (shared across services),
  `auth-service`, and client SDKs — this is the scenario most likely to require touching several
  repos/services, not one file.
- **Kubernetes Action Required:** Rolling restarts of `auth-service`, `checkout-service`, and any
  other service using the shared cache-fallback library, after each fix; during the live incident
  itself (before root-causing), the *correct* immediate mitigation is enabling a circuit breaker /
  shedding load at the gateway, not restarting pods (restarting mid-stampede does not help and may
  extend the outage by discarding warm connections).
- **Database Changes Required:** Recommended — a dedicated read-replica or connection-pool
  partition for session-lookup fallback traffic, isolated from primary transactional traffic.
- **Configuration Changes Required:** Yes — circuit-breaker thresholds, retry/backoff/jitter
  settings, dedicated pool sizing.
- **Postman APIs Expected to Fail (pre-fix):** Effectively all authenticated endpoints across all
  services during a simulated Redis failover (`GET /api/me`, `POST /api/checkout`,
  `POST /api/cart/items`, `GET /api/orders`) — the collection for this scenario should be run
  *during* a chaos-injected Redis restart, not against a steady-state system.
- **Expected Final Outcome:** **Multi-iteration**, and the scenario expects the agent to correctly
  distinguish immediate mitigation (circuit-break/shed load) from root-cause remediation (the
  three code/config fixes above) across at least two full remediation cycles — first pass
  stabilizes the incident (stampede protection in the shared cache client), second pass addresses
  the residual auth-service pool contention surfaced only once the stampede itself is fixed and
  the "cached sessions going stale under sustained load" pattern becomes visible.
- **Success Criteria:** A simulated 30s Redis failover under representative load results in a
  measurable, brief (<45s) latency blip with zero 5xx amplification and full recovery within 2
  minutes, not 25; DB connection count during the simulated failover stays within its allocated
  budget; client retry telemetry shows backoff/jitter behaving as configured (no synchronized
  retry spikes).

---

## Extending this library

Categories from the original brief not yet given a dedicated scenario above — Azure/GCP
credential failures, DNS failures, disk-full, high CPU (ReDoS-style), circular imports, infinite
loops, corrupted data, git merge conflicts blocking the auto-push step, CI/CD pipeline failures,
Helm chart schema mismatches, and Docker build failures — follow the same template and are
natural next additions. When adding one, keep the same field set (ID, Difficulty, Failure
Category, Initial Symptoms, Sample Error Logs, Root Cause, Expected Files to Modify, Whether Code
Changes Are Required, Kubernetes Action Required, Database Changes Required, Configuration Changes
Required, Postman APIs Expected to Fail, Expected Final Outcome, Success Criteria) and make sure
roughly a third of new scenarios are deliberately multi-iteration, mirroring the mix above.
