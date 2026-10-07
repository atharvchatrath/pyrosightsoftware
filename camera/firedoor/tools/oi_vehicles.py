"""List Open Images images that show a vehicle (car, bus, train, plane, boat, ...): data/oi/vehicle_ids.json.

Open Images also boxes car, bus, train and plane windows as "Window". Those are not a way out of a
building, so build_index.py leaves window photos of vehicles out of training and evaluation.
Train images: the human-verified labels in train-il-window.csv / train-il-filtered.csv (Car, Vehicle,
Land vehicle, Bus, Train; see oi_train_stream.py). Validation and test images: all their human-verified
labels and boxes.
"""
import csv, json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'oi')
NAMES = ['Car', 'Vehicle', 'Land vehicle', 'Bus', 'Train', 'Truck', 'Van', 'Taxi', 'Airplane', 'Aircraft',
         'Boat', 'Limousine', 'Motorcycle', 'Ambulance', 'Vehicle registration plate', 'Wheel', 'Tire',
         'Auto part', 'Golf cart', 'Tank', 'Helicopter', 'Snowmobile', 'Cart']
mid = {}
for l in open(os.path.join(D, 'cd.csv')):
    k, v = l.rstrip('\n').split(',', 1)
    mid[k] = v.strip('"')
V = {k for k, v in mid.items() if v in NAMES}
ids = set()
for f, col, conf in [('train-il-window.csv', 2, 3), ('train-il-filtered.csv', 2, 3),
                     ('validation-annotations-human-imagelabels.csv', 2, 3), ('test-annotations-human-imagelabels.csv', 2, 3),
                     ('validation-annotations-bbox.csv', 2, None), ('test-annotations-bbox.csv', 2, None)]:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print('missing', f)
        continue
    n = 0
    for row in csv.reader(open(p)):
        if row[col] in V and (conf is None or row[conf] == '1'):
            ids.add(row[0]); n += 1
    print(f, n, 'vehicle rows')
json.dump(sorted(ids), open(os.path.join(D, 'vehicle_ids.json'), 'w'))
print(len(ids), 'images with a vehicle ->', os.path.join(D, 'vehicle_ids.json'))
