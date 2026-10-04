"""Sanity-check the Keras->PyTorch MobileNetV2 weight mapping with the ImageNet classifier top:
crop large Open Images boxes of classes that exist in ImageNet and measure top-5 accuracy."""
import json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import MobileNetV2, load_keras_mnv2
torch.set_num_threads(2)
R = os.path.join(os.path.dirname(__file__), '..')
idx = json.load(open(os.path.join(R, 'data/imagenet_class_index.json')))
names = {int(k): v[1] for k, v in idx.items()}
want = {'Orange': {950}, 'Traffic light': {920}, 'Television': {851, 664, 782}, 'Computer monitor': {664, 782, 851, 527},
        'Pumpkin': {607, 988}, 'Lamp': {846, 619}, 'Christmas tree': {}, 'Street light': {}}
sel = json.load(open(os.path.join(R, 'data/oi/selection.json')))
crops = []
for v in sel:
    p = os.path.join(R, 'data/oi/img', v['id'] + '.jpg')
    if not os.path.exists(p):
        continue
    for f in v['focus']:
        if want.get(f[0]) and (f[3] - f[1]) * (f[4] - f[2]) > 0.08:
            crops.append((p, f)); break
crops = crops[:300]
def prep(p, f):
    im = Image.open(p).convert('RGB'); W, H = im.size
    im = im.crop((f[1] * W, f[2] * H, f[3] * W, f[4] * H)).resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32).transpose(2, 0, 1) / 127.5 - 1
X = torch.from_numpy(np.stack([prep(p, f) for p, f in crops]))
Y = [want[f[0]] for p, f in crops]
print('crops', len(crops))
for label, kp, load in [('keras-asym-pad', True, True), ('symmetric-pad', False, True), ('random-init', False, False)]:
    m = MobileNetV2(0.5, with_top=True)
    if load:
        load_keras_mnv2(m, os.path.join(R, 'data/mnv2_0.5_224_top.h5'))
    m.set_keras_pad(kp); m.eval()
    with torch.no_grad():
        top5 = m(X).topk(5, 1).indices.numpy()
    acc1 = np.mean([t[0] in y for t, y in zip(top5, Y)]); acc5 = np.mean([len(set(t) & y) > 0 for t, y in zip(top5, Y)])
    print(f'{label:16s} top1 {acc1:.3f} top5 {acc5:.3f}  e.g. {[names[i] for i in top5[0][:3]]} for {crops[0][1][0]}')
