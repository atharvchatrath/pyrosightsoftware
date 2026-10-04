// Node counterpart of tensorfile.py
const fs = require('fs');
function load(prefix) {
  const meta = JSON.parse(fs.readFileSync(prefix + '.json', 'utf8'));
  const buf = fs.readFileSync(prefix + '.bin');
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const arrays = {};
  for (const a of meta.arrays) {
    const n = a.shape.reduce((x, y) => x * y, 1);
    arrays[a.name] = { shape: a.shape, data: new Float32Array(ab, a.offset, n) };
  }
  return { meta, arrays };
}
function save(prefix, arrays, extra) {
  const meta = Object.assign({ arrays: [] }, extra || {});
  const parts = [];
  let off = 0;
  for (const [name, shape, data] of arrays) {
    const f = Float32Array.from(data);
    meta.arrays.push({ name, shape, offset: off });
    parts.push(Buffer.from(f.buffer));
    off += f.byteLength;
  }
  fs.writeFileSync(prefix + '.json', JSON.stringify(meta));
  fs.writeFileSync(prefix + '.bin', Buffer.concat(parts));
}
module.exports = { load, save };
