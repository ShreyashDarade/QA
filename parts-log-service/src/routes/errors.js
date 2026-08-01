const express = require('express');
const { v4: uuid } = require('uuid');
const { Collection } = require('../store');

const errors = new Collection('errors');
const router = express.Router();

/**
 * Errors can also be reported externally (e.g. from another service, a log
 * shipper, or a monitoring hook) rather than only caught in-process by
 * errorLogger middleware. Either path lands in the same collection and is
 * picked up by the AI fixer the same way - capture is plain, deterministic
 * bookkeeping. No AI runs here.
 */
router.post('/', (req, res) => {
  const { partKey, message, stack, source, metadata } = req.body || {};

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'message (string) is required' });
  }

  const entry = {
    id: uuid(),
    partKey: partKey || null,
    message,
    stack: stack || null,
    source: source || 'external',
    metadata: metadata || {},
    status: 'new', // new -> fixing -> fixed | fix_failed
    fixAttempts: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };

  errors.insert(entry);
  res.status(201).json(entry);
});

router.get('/', (req, res) => {
  const { status, partKey } = req.query;
  let results = errors.all();
  if (status) results = results.filter((e) => e.status === status);
  if (partKey) results = results.filter((e) => e.partKey === partKey);
  res.json(results);
});

router.get('/:id', (req, res) => {
  const entry = errors.getById(req.params.id);
  if (!entry) return res.status(404).json({ error: 'not found' });
  res.json(entry);
});

module.exports = { router, errors };
