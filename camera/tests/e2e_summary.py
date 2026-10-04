#!/usr/bin/env python3
"""Markdown tables from tests/out/e2e_results.json (for README.md).
    python3 tests/e2e_summary.py [results.json]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out', 'e2e_results.json')))
order = ['faces', 'group', 'nopeople', 'fire', 'firevideo', 'doors', 'lights']
desc = {'faces': '8 close-up people photos', 'group': '8 photos, 3-6 people', 'nopeople': '8 photos without people',
        'fire': '12 FireNET test fire photos', 'firevideo': 'CCTV clip, fire in every frame', 'doors': '10 door photos',
        'lights': '12 lamps, bulbs, sunsets, oranges...'}


def frac(a, b):
    return '%d/%d' % (a, b) if b else '-'


print('| Backend | Clip | Updates | Median update (p90) | people / faces / fire+door ms | Tensors | WHITE | PURPLE | GREEN | Requests, errors |')
print('|---|---|---|---|---|---|---|---|---|---|')
for be in ('webgl', 'cpu'):
    for n in order:
        x = r['runs'].get('%s_%s' % (be, n))
        if not x or x.get('error'):
            if x:
                print('| %s | %s | ERROR %s |' % (be, n, x['error'].splitlines()[0]))
            continue
        s, t = x['score'], x['timing']
        p, f, d = s['person'], s['fire'], s['door']
        if p['gtUpdates']:
            white = 'person boxed in %s updates; %s GT people found; %d/%d boxes off any person' % (frac(p['hit'], p['gtUpdates']), frac(p['personsFound'], p['gtPersons']), p['unmatchedBoxes'], p['boxes'])
        elif n == 'nopeople':
            white = 'boxes in %s updates' % frac(sum(1 for q in x['records'] if q['people']), s['updates'])
        else:
            white = '(not scored)'
        if f['gtUpdates']:
            purple = 'FIRE in %s updates' % frac(f['any'], f['gtUpdates']) + ('' if n == 'firevideo' else ', on the fire %s' % frac(f['hit'], f['gtUpdates']))
        else:
            purple = 'false FIRE in %s updates (%d boxes)' % (frac(f['falseUpdates'], f['noGtUpdates']), f['falseBoxes'])
        green = ('DOOR in %s updates, on a door %s' % (frac(d['any'], d['gtUpdates']), frac(d['hit'], d['gtUpdates']))) if d['gtUpdates'] else ('%d DOOR boxes (no door ground truth)' % d['otherBoxes'])
        be_s = x['start']['backend'] + (' (software)' if x['start'].get('backendInfo', {}).get('software') else '')
        print('| %s | %s: %s | %d | %s ms (%s) | %s / %s / %s | %s..%s | %s | %s | %s | %d, %d |' % (
            be_s, n, desc[n], s['updates'], t['totalMedian'], t['totalP90'], t['personMedian'], t['faceMedian'], t['firedoorMedian'],
            x['tensors']['min'], x['tensors']['max'], white, purple, green, len(x['external']), len(x['pageErrors'])))
print()
for k, x in r['runs'].items():
    if x.get('actions'):
        print('%s actions: %s' % (k, json.dumps(x['actions'])))
    if x.get('start'):
        print('%s: load %s ms, first update %s ms after Start; status: %s' % (k, x['start']['loadMs'], x.get('firstUpdateAfterStartMs'), x['final']['status']['speed'] if x.get('final') else ''))
