const express = require('express');
const { v4: uuid } = require('uuid');
const { Collection } = require('../store');
const { registry } = require('./registry');

const parts = new Collection('parts');
const router = express.Router();

/**
 * Universal parts log. partKey is supplied entirely by the caller - it does
 * not need to pre-exist in the registry. Unknown keys are still logged (and
 * flagged as unregistered) so nothing gets silently dropped just because a
 * new part name showed up before anyone registered it.
 */
router.post('/', (req, res) => {
  const { partKey, event, level, message, metadata } = req.body || {};

  if (!partKey || typeof partKey !== 'string') {
    return res.status(400).json({ error: 'partKey (string) is required' });
  }
  if (!event || typeof event !== 'string') {
    return res.status(400).json({ error: 'event (string) is required' });
  }

  const known = registry.findOne((r) => r.key === partKey || (r.aliases || []).includes(partKey));

  const entry = {
    id: uuid(),
    partKey,
    registeredKey: known ? known.key : null,
    event,
    level: level || 'info',
    message: message || '',
    metadata: metadata || {},
    createdAt: new Date().toISOString(),
  };

  parts.insert(entry);
  res.status(201).json(entry);
});

router.get('/', (req, res) => {
  const { partKey, level, since } = req.query;
  let results = parts.all();

  if (partKey) results = results.filter((p) => p.partKey === partKey);
  if (level) results = results.filter((p) => p.level === level);
  if (since) {
    const sinceDate = new Date(since);
    results = results.filter((p) => new Date(p.createdAt) >= sinceDate);
  }

  res.json(results);
});

router.get('/:id', (req, res) => {
  const entry = parts.getById(req.params.id);
  if (!entry) return res.status(404).json({ error: 'not found' });
  res.json(entry);
});

module.exports = { router, parts };
