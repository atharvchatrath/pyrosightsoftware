#!/usr/bin/env node
// Person-box agreement between two run_people_corpus.js outputs.
//   node agree.js REF.json CAND.json [thr...]
const fs = require('fs');
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const i = ix * iy;
  return i / (a.w * a.h + b.w * b.h - i);
}
function agree(A, B, thr) {
  const bm = {};
  B.images.forEach((i) => { bm[i.file] = i; });
  let na = 0, nb = 0, m = 0, s = 0, mi = 1, md = 0, sd = 0;
  const ua = [], ub = [];
  for (const ia of A.images) {
    const ib = bm[ia.file];
    if (!ib) continue;
    const pa = ia.persons.filter((d) => d.score >= thr), pb = ib.persons.filter((d) => d.score >= thr);
    na += pa.length; nb += pb.length;
    const used = new Set();
    for (const d of pa) {
      let best = -1, bi = 0.5;
      pb.forEach((q, j) => { if (!used.has(j)) { const v = iou(d, q); if (v > bi) { bi = v; best = j; } } });
      if (best < 0) { ua.push(+d.score.toFixed(3)); continue; }
      used.add(best); m++; s += bi; mi = Math.min(mi, bi);
      const ds = Math.abs(d.score - pb[best].score);
      md = Math.max(md, ds); sd += ds;
    }
    pb.forEach((q, j) => { if (!used.has(j)) ub.push(+q.score.toFixed(3)); });
  }
  return { thr, ref: na, cand: nb, matched: m, meanIoU: +(s / m).toFixed(4), minIoU: +mi.toFixed(3),
    meanScoreDiff: +(sd / m).toFixed(4), maxScoreDiff: +md.toFixed(4), unmatchedRef: ua, unmatchedCand: ub };
}
const [a, b, ...t] = process.argv.slice(2);
const A = JSON.parse(fs.readFileSync(a, 'utf8')), B = JSON.parse(fs.readFileSync(b, 'utf8'));
for (const thr of (t.length ? t.map(Number) : [0.4, 0.5])) console.log(JSON.stringify(agree(A, B, thr)));
