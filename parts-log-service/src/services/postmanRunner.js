const fs = require('fs');

/**
 * Executes a standard Postman Collection v2.1 JSON export against a base
 * URL, with no dependency on the Postman app or the `newman` CLI - just
 * fetch. Deliberately simple: pass/fail is "did the response come back
 * with a non-error status", which is enough to catch "still 500ing" /
 * "still 404ing" style regressions without needing to reimplement
 * Postman's full test-script sandbox.
 *
 * If POSTMAN_COLLECTION_PATH isn't configured, this step is skipped and
 * treated as passing (not failing) so the rest of the loop still runs
 * end-to-end without requiring a collection file.
 */

function flattenItems(items, prefix = []) {
  const out = [];
  for (const item of items || []) {
    if (item.item) {
      out.push(...flattenItems(item.item, [...prefix, item.name]));
    } else if (item.request) {
      out.push({ ...item, folderPath: prefix });
    }
  }
  return out;
}

function resolveUrl(request, baseUrl) {
  let raw = typeof request.url === 'string' ? request.url : (request.url && request.url.raw) || '';
  if (baseUrl) {
    raw = raw.replace(/\{\{\s*baseUrl\s*\}\}/gi, baseUrl);
  }
  return raw;
}

function buildHeaders(request) {
  const headers = {};
  for (const h of request.header || []) {
    if (!h.disabled) headers[h.key] = h.value;
  }
  return headers;
}

async function runCollection({ collectionPath, baseUrl, filter }) {
  if (!collectionPath || !fs.existsSync(collectionPath)) {
    return {
      skipped: true,
      allPassed: true,
      results: [],
      note: 'No Postman collection configured - step skipped.',
    };
  }

  const collection = JSON.parse(fs.readFileSync(collectionPath, 'utf8'));
  let items = flattenItems(collection.item);

  if (filter) {
    const matched = items.filter((it) =>
      `${it.name} ${resolveUrl(it.request, baseUrl)}`
        .toLowerCase()
        .includes(String(filter).toLowerCase())
    );
    // Narrow to the "affected" requests when we can identify them by name/
    // URL; fall back to the full collection otherwise.
    if (matched.length) items = matched;
  }

  const results = [];
  for (const item of items) {
    const method = (item.request.method || 'GET').toUpperCase();
    const url = resolveUrl(item.request, baseUrl);
    const headers = buildHeaders(item.request);
    const body =
      item.request.body && item.request.body.mode === 'raw' && !['GET', 'HEAD'].includes(method)
        ? item.request.body.raw
        : undefined;

    try {
      const res = await fetch(url, { method, headers, body });
      results.push({ name: item.name, method, url, status: res.status, ok: res.status < 400 });
    } catch (err) {
      results.push({ name: item.name, method, url, status: null, ok: false, error: err.message });
    }
  }

  return { skipped: false, allPassed: results.every((r) => r.ok), results };
}

module.exports = { runCollection, flattenItems, resolveUrl };
