#!/usr/bin/env node
// Evaluate person / face / merged-display detections from run_people_corpus.js
// against testdata/people/manifest.json (Open Images boxes) and mediapipe.json.
//   node eval_people.js FULL_RUN.json [ORIG_RUN.json] [--json out.json]
const fs = require('fs');
const path = require('path');
require('../people.js');

const TD = path.join(__dirname, '../../testdata/people');

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const i = ix * iy;
  const u = a.w * a.h + b.w * b.h - i;
  return u > 0 ? i / u : 0;
}
function ioa(a, b) { // fraction of a inside b
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (ix * iy) / Math.max(1e-12, a.w * a.h);
}
const B = (v) => ({ x: v[0], y: v[1], w: v[2], h: v[3] });
const center = (b) => [b.x + b.w / 2, b.y + b.h / 2];
const inside = (p, b) => p[0] >= b.x && p[0] <= b.x + b.w && p[1] >= b.y && p[1] <= b.y + b.h;
const faceMatch = (d, g) => iou(d, g) >= 0.3 || (inside(center(d), g) && inside(center(g), d));

function personPR(images, gt, thr, filt) {
  let tp = 0, fp = 0, ngt = 0, ngtBig = 0, tpBig = 0;
  const byGroup = {};
  let negImgsWithFP = 0, negImgs = 0, posImgsHit = 0, posImgs = 0;
  for (const im of images) {
    const g = gt[im.file];
    if (!g || (filt && !filt(g))) continue;
    const gts = g.persons.map(B), ign = g.ignore.map(B);
    const dets = im.persons.filter((d) => d.score >= thr).sort((a, b) => b.score - a.score);
    const used = new Set();
    let imgFP = 0, imgTP = 0;
    for (const d of dets) {
      let best = -1, bi = 0.5;
      gts.forEach((q, j) => { if (!used.has(j)) { const v = iou(d, q); if (v >= bi) { bi = v; best = j; } } });
      if (best >= 0) { used.add(best); tp++; imgTP++; if (gts[best].h >= 0.2) tpBig++; continue; }
      if (ign.some((q) => iou(d, q) >= 0.5 || ioa(d, q) >= 0.5)) continue;   // group/depiction regions
      fp++; imgFP++;
    }
    ngt += gts.length;
    ngtBig += gts.filter((q) => q.h >= 0.2).length;
    const s = byGroup[g.group] = byGroup[g.group] || { tp: 0, gt: 0, fp: 0, imgs: 0 };
    s.tp += imgTP; s.gt += gts.length; s.fp += imgFP; s.imgs++;
    if (g.group === 'negative') { negImgs++; if (imgFP) negImgsWithFP++; } else { posImgs++; if (dets.length) posImgsHit++; }
  }
  for (const k in byGroup) byGroup[k].recall = byGroup[k].gt ? byGroup[k].tp / byGroup[k].gt : null;
  return { thr, tp, fp, gt: ngt, recall: tp / ngt, precision: tp / Math.max(1, tp + fp),
    recallBig: tpBig / ngtBig, gtBig: ngtBig, negImagesWithPersonFP: negImgsWithFP / Math.max(1, negImgs),
    posImagesWithAnyPerson: posImgsHit / Math.max(1, posImgs), byGroup };
}

function faceStats(images, gt, thr) {
  let closeups = 0, hit = 0, negFP = 0, negImgs = 0, allGT = 0, allHit = 0, fpFaces = 0, nf = 0;
  for (const im of images) {
    const g = gt[im.file];
    if (!g) continue;
    const dets = (im.faces || []).filter((d) => d.score >= thr);
    if (g.group === 'negative') { negImgs++; if (dets.length) negFP++; continue; }
    const gf = g.faces.map(B);
    const persons = g.persons.map(B);
    const used = new Set();
    gf.forEach((q) => { const j = dets.findIndex((d, k) => !used.has(k) && faceMatch(d, q)); if (j >= 0) { used.add(j); allHit++; } });
    allGT += gf.length;
    nf += dets.length;
    // a detected face that matches no GT face and lies in no GT person box
    dets.forEach((d, k) => { if (!used.has(k) && !persons.some((p) => ioa(d, p) >= 0.6) && !gf.some((q) => ioa(d, q) >= 0.5)) fpFaces++; });
    if (g.group === 'closeup') {
      closeups++;
      if (gf.length && dets.some((d) => faceMatch(d, gf[0]))) hit++;
    }
  }
  return { thr, closeupFaceRecall: hit / closeups, closeups, faceRecallAllGroups: allHit / allGT, gtFaces: allGT,
    negImagesWithFace: negFP / Math.max(1, negImgs), facesNotOnAnyPerson: fpFaces, faces: nf };
}

function displayStats(images, gt, pthr, fthr) {
  const o = { personMinScore: pthr, faceMinScore: fthr };
  let closeups = 0, closeWhite = 0, closeWhiteByPerson = 0, closeWhiteByFaceOnly = 0;
  let tp = 0, fp = 0, dup = 0, n = 0, negBoxes = 0, negImgsWithBox = 0, negImgs = 0;
  let personsCovered = 0, personsGT = 0, faceDist = 0, nPersonBoxes = 0;
  for (const im of images) {
    const g = gt[im.file];
    if (!g) continue;
    const dets = im.persons.filter((d) => d.score >= pthr).map((d) => Object.assign({ cls: 'person' }, d))
      .concat((im.faces || []).filter((d) => d.score >= Math.min(fthr, PSPeople.DEFAULTS.faceAttachMinScore)).map((d) => Object.assign({ cls: 'face' }, d)));
    const disp = PSPeople.merge(dets, im.width, im.height, o);
    n += disp.length;
    disp.forEach((d) => { if (d.src !== 'face') { nPersonBoxes++; if (d.src === 'person+face') faceDist++; } });
    if (g.group === 'negative') { negImgs++; negBoxes += disp.length; if (disp.length) negImgsWithBox++; continue; }
    const gp = g.persons.map(B), gf = g.faces.map(B), ign = g.ignore.map(B);
    // display box correctness
    const usedP = new Set();
    for (const d of disp.slice().sort((a, b) => b.score - a.score)) {
      let best = -1, bi = 0.5;
      gp.forEach((q, j) => { if (!usedP.has(j)) { const v = iou(d, q); if (v >= bi) { bi = v; best = j; } } });
      if (best >= 0) { usedP.add(best); tp++; continue; }
      if (d.src === 'face') {
        const host = gp.findIndex((q) => ioa(d, q) >= 0.6);
        if (host >= 0 && usedP.has(host)) { dup++; continue; }            // person already boxed
        if (host >= 0 || gf.some((q) => faceMatch(d.face || d, q))) { if (host >= 0) usedP.add(host); tp++; continue; }
      }
      if (ign.some((q) => iou(d, q) >= 0.5 || ioa(d, q) >= 0.5)) continue;
      fp++;
    }
    personsGT += gp.length;
    personsCovered += gp.filter((q) => disp.some((d) => iou(d, q) >= 0.5 || (d.src === 'face' && ioa(d, q) >= 0.6))).length;
    if (g.group === 'closeup' && gf.length) {
      closeups++;
      const f = gf[0];
      const covering = disp.filter((d) => ioa(f, d) >= 0.5 || (d.face && faceMatch(d.face, f)));
      if (covering.length) {
        closeWhite++;
        if (covering.some((d) => d.src !== 'face')) closeWhiteByPerson++; else closeWhiteByFaceOnly++;
      }
    }
  }
  return { personThr: pthr, faceThr: fthr, closeupsWithWhiteBox: closeWhite / closeups, closeups,
    personBoxesWithFaceDistance: faceDist, personBoxes: nPersonBoxes,
    closeupWhiteFromPersonModel: closeWhiteByPerson, closeupWhiteOnlyFromFace: closeWhiteByFaceOnly,
    displayPrecision: tp / Math.max(1, tp + fp), displayTP: tp, displayFP: fp, displayDuplicates: dup,
    personCoverage: personsCovered / personsGT, negImagesWithBox: negImgsWithBox / Math.max(1, negImgs),
    negBoxes };
}

function mediapipeImages(images, pthr, fthr) {
  const mp = JSON.parse(fs.readFileSync(path.join(TD, 'mediapipe.json'), 'utf8')).items;
  const out = [];
  for (const it of mp) {
    const im = images.find((q) => q.file === it.file);
    if (!im) continue;
    const dets = im.persons.filter((d) => d.score >= pthr).map((d) => Object.assign({ cls: 'person' }, d))
      .concat((im.faces || []).filter((d) => d.score >= Math.min(fthr, PSPeople.DEFAULTS.faceAttachMinScore)).map((d) => Object.assign({ cls: 'face' }, d)));
    const disp = PSPeople.merge(dets, im.width, im.height, { personMinScore: pthr, faceMinScore: fthr });
    out.push({ file: it.file, expectPersons: it.persons, partial: !!it.partial,
      persons: dets.filter((d) => d.cls === 'person').length, faces: dets.filter((d) => d.cls === 'face').length,
      display: disp.length, displaySrc: disp.map((d) => d.src), note: it.note });
  }
  return out;
}

function agreement(a, b, thr) {
  // a = reference (orig), b = candidate; per-image greedy match of persons >= thr
  let na = 0, nb = 0, m = 0, iouSum = 0, iouMin = 1, maxDs = 0;
  const unA = [], unB = [];
  const bm = {};
  for (const im of b.images) bm[im.file] = im;
  for (const ia of a.images) {
    const ib = bm[ia.file];
    if (!ib) continue;
    const pa = ia.persons.filter((d) => d.score >= thr), pb = ib.persons.filter((d) => d.score >= thr);
    na += pa.length; nb += pb.length;
    const used = new Set();
    for (const d of pa) {
      let best = -1, bi = 0.5;
      pb.forEach((q, j) => { if (!used.has(j)) { const v = iou(d, q); if (v > bi) { bi = v; best = j; } } });
      if (best < 0) { unA.push(+d.score.toFixed(3)); continue; }
      used.add(best); m++; iouSum += bi; iouMin = Math.min(iouMin, bi);
      maxDs = Math.max(maxDs, Math.abs(d.score - pb[best].score));
    }
    pb.forEach((q, j) => { if (!used.has(j)) unB.push(+q.score.toFixed(3)); });
  }
  return { thr, ref: na, cand: nb, matched: m, meanIoU: iouSum / Math.max(1, m), minIoU: iouMin, maxScoreDiff: maxDs,
    unmatchedRefScores: unA, unmatchedCandScores: unB };
}

function main() {
  const args = process.argv.slice(2);
  const ji = args.indexOf('--json');
  const outPath = ji >= 0 ? args.splice(ji, 2)[1] : null;
  const run = JSON.parse(fs.readFileSync(args[0], 'utf8'));
  const orig = args[1] ? JSON.parse(fs.readFileSync(args[1], 'utf8')) : null;
  const man = JSON.parse(fs.readFileSync(path.join(TD, 'manifest.json'), 'utf8'));
  const gt = {};
  for (const it of man.items) gt[it.file] = it;
  const R = { run: run.variant, medianMs: run.medianMs, tensors: [run.numTensorsAfterFirst, run.numTensorsEnd] };
  const ths = [0.3, 0.4, 0.5, 0.6, 0.7];
  R.person = ths.map((t) => personPR(run.images, gt, t));
  if (orig) {
    R.personOrig = ths.map((t) => personPR(orig.images, gt, t));
    R.agreement = [0.3, 0.5].map((t) => agreement(orig, run, t));
  }
  if (run.images[0].faces) {
    R.face = [0.5, 0.6, 0.7, 0.75, 0.8].map((t) => faceStats(run.images, gt, t));
    R.display = [];
    for (const p of [0.3, 0.4, 0.5, 0.6]) for (const f of [0.5, 0.6, 0.7, 0.75, 0.8]) R.display.push(displayStats(run.images, gt, p, f));
    R.mediapipe = mediapipeImages(run.images, PSPeople.DEFAULTS.personMinScore, PSPeople.DEFAULTS.faceMinScore);
  }
  const s = JSON.stringify(R, null, 1);
  if (outPath) fs.writeFileSync(outPath, s);
  // compact print
  const f = (v) => (typeof v === 'number' ? +v.toFixed(3) : v);
  console.log('PERSON (repacked)  thr  recall  recall(h>=.2)  precision  negImgFP  byGroup recall');
  for (const r of R.person) console.log('  ', r.thr, f(r.recall), f(r.recallBig), f(r.precision), f(r.negImagesWithPersonFP),
    Object.entries(r.byGroup).map(([k, v]) => k + ':' + f(v.recall) + '/fp' + v.fp).join(' '));
  if (R.personOrig) {
    console.log('PERSON (original coco-ssd float32)');
    for (const r of R.personOrig) console.log('  ', r.thr, f(r.recall), f(r.recallBig), f(r.precision), f(r.negImagesWithPersonFP));
    for (const a of R.agreement) console.log('AGREE', JSON.stringify(a));
  }
  if (R.face) {
    console.log('FACE thr closeupRecall allRecall negImgFace facesNotOnPerson');
    for (const r of R.face) console.log('  ', r.thr, f(r.closeupFaceRecall), f(r.faceRecallAllGroups), f(r.negImagesWithFace), r.facesNotOnAnyPerson);
    console.log('DISPLAY pThr fThr closeupWhite(byPerson/faceOnly) precision TP FP dup coverage negImgBox');
    for (const r of R.display) console.log('  ', r.personThr, r.faceThr, f(r.closeupsWithWhiteBox), '(' + r.closeupWhiteFromPersonModel + '/' + r.closeupWhiteOnlyFromFace + ')',
      f(r.displayPrecision), r.displayTP, r.displayFP, r.displayDuplicates, f(r.personCoverage), f(r.negImagesWithBox),
      'faceDist ' + r.personBoxesWithFaceDistance + '/' + r.personBoxes);
    console.log('MEDIAPIPE images (defaults)');
    for (const m of R.mediapipe) console.log('  ', m.file, 'expect', m.expectPersons, 'persons', m.persons, 'faces', m.faces, 'display', m.display, m.displaySrc.join(','), m.partial ? '(partial)' : '');
  }
}
main();
