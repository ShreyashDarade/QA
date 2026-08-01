const fs = require('fs');
const path = require('path');
const config = require('./config');

/**
 * Minimal append-friendly JSON-file collection. Deliberately dependency-free
 * so this service runs with zero external infrastructure out of the box;
 * swap for Postgres/Mongo/etc. later by re-implementing this same interface.
 */
class Collection {
  constructor(name) {
    this.file = path.join(config.dataDir, `${name}.json`);
    this._ensure();
  }

  _ensure() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    if (!fs.existsSync(this.file)) {
      fs.writeFileSync(this.file, '[]');
    }
  }

  _readAll() {
    this._ensure();
    const raw = fs.readFileSync(this.file, 'utf8').trim();
    return raw ? JSON.parse(raw) : [];
  }

  _writeAll(records) {
    // Atomic-ish write: write to a temp file then rename, so a crash
    // mid-write can't corrupt the store.
    const tmp = `${this.file}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(records, null, 2));
    fs.renameSync(tmp, this.file);
  }

  all() {
    return this._readAll();
  }

  find(predicate) {
    return this._readAll().filter(predicate);
  }

  findOne(predicate) {
    return this._readAll().find(predicate) || null;
  }

  getById(id) {
    return this.findOne((r) => r.id === id);
  }

  insert(record) {
    const records = this._readAll();
    records.push(record);
    this._writeAll(records);
    return record;
  }

  update(id, patch) {
    const records = this._readAll();
    const idx = records.findIndex((r) => r.id === id);
    if (idx === -1) return null;
    records[idx] = { ...records[idx], ...patch, updatedAt: new Date().toISOString() };
    this._writeAll(records);
    return records[idx];
  }
}

module.exports = { Collection };
