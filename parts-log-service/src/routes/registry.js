const express = require('express');
const { v4: uuid } = require('uuid');
const { Collection } = require('../store');

const registry = new Collection('registry');
const router = express.Router();

/**
 * The registry has no fixed list of part names or "kinds" - any caller can
 * register a new part key at any time. This is what makes parts, key
 * commands, aliases, etc. externally addable instead of hardcoded.
 *
 * A registry entry maps a partKey to the source file the AI fixer is
 * allowed to patch for that part, plus free-form metadata (aliases / key
 * commands / owners / anything the caller wants to attach).
 */
router.post('/', (req, res) => {
  const { key, file, description, aliases, metadata } = req.body || {};

  if (!key || typeof key !== 'string') {
    return res.status(400).json({ error: 'key (string) is required' });
  }

  const existing = registry.findOne((r) => r.key === key);
  if (existing) {
    const updated = registry.update(existing.id, {
      file: file ?? existing.file,
      description: description ?? existing.description,
      aliases: aliases ?? existing.aliases,
      metadata: { ...existing.metadata, ...(metadata || {}) },
    });
    return res.status(200).json(updated);
  }

  const entry = {
    id: uuid(),
    key,
    file: file || null,
    description: description || '',
    aliases: Array.isArray(aliases) ? aliases : [],
    metadata: metadata || {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  registry.insert(entry);
  res.status(201).json(entry);
});

router.get('/', (req, res) => {
  res.json(registry.all());
});

router.get('/:key', (req, res) => {
  const entry = registry.findOne(
    (r) => r.key === req.params.key || (r.aliases || []).includes(req.params.key)
  );
  if (!entry) return res.status(404).json({ error: 'not found' });
  res.json(entry);
});

module.exports = { router, registry };
