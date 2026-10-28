"""Independent GLB/FK, preservation and constraint checks on own-prompt outputs.
Development only; standard library. Usage: python tests/momask-authoring.py FOLDER
The folder contains generated GLBs/recipes, not upstream evaluation data.
"""
import json
import hashlib
import math
from pathlib import Path
import struct
import sys
from soma_metrics import payload


def tracks(path):
    doc, binary = payload(path)
    result = {}
    for channel in doc['animations'][0]['channels']:
        target = channel['target']; sampler = doc['animations'][0]['samplers'][channel['sampler']]
        accessor = doc['accessors'][sampler['output']]; view = doc['bufferViews'][accessor['bufferView']]
        width = {'VEC3': 3, 'VEC4': 4}[accessor['type']]
        offset = view.get('byteOffset', 0)+accessor.get('byteOffset', 0)
        result[doc['nodes'][target['node']]['name'], target['path']] = [
            struct.unpack_from('<'+'f'*width, binary, offset+i*width*4) for i in range(accessor['count'])]
    return doc, result


def mul(a, b):
    x,y,z,w = a; X,Y,Z,W = b
    return w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z


def world(doc, data, frame, target=None):
    nodes = doc['nodes']; parents = {c:i for i,n in enumerate(nodes) for c in n.get('children', [])}
    result = {}
    def visit(i):
        if i in result: return result[i]
        name = nodes[i]['name']
        p = data[name,'translation'][frame]
        q = data[name,'rotation'][frame]
        if target:
            q = target['targetRotations'][i]
            p = target['targetRoot'] if i == 0 else nodes[i]['translation']
        if i in parents:
            pp,pq = visit(parents[i]); rotated = mul(mul(pq,(*p,0)),(-pq[0],-pq[1],-pq[2],pq[3]))
            p = tuple(pp[k]+rotated[k] for k in range(3)); q = mul(pq,q)
        result[i] = p,q
        return p,q
    return [visit(i) for i in range(len(nodes))]


def angle(a, b):
    dot = abs(sum(x*y for x,y in zip(a,b)))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return math.degrees(2*math.acos(min(1,dot)))


def delta(a, b, rotation):
    if rotation and sum(x*y for x,y in zip(a,b)) < 0: b = tuple(-x for x in b)
    return max(abs(x-y) for x,y in zip(a,b))


folder = Path(sys.argv[1]); body = None
for path in folder.glob('*.glb'):
    doc, data = tracks(path)
    if len(doc['nodes']) == 30: body = set(data); break
assert body, 'Include a canonical SOMA-30 generation for the independent mapping'
measurements = []
for path in sorted(folder.glob('*.glb')):
    doc, data = tracks(path); n = len(next(iter(data.values())))
    assert all(len(rows) == n for rows in data.values())
    assert all(math.isfinite(x) for rows in data.values() for row in rows for x in row)
    for rows in data.values():
        if len(rows[0]) == 4:
            assert all(abs(sum(x*x for x in q)-1) < 2e-5 for q in rows)
    recipe_path = path.with_suffix('.glb.json')
    if not recipe_path.exists(): continue
    recipe = json.loads(recipe_path.read_text())
    for source in recipe.get('poseSources',[]):
        assert hashlib.sha256(Path(source['file']).read_bytes()).hexdigest() == source['sha256']
    assert n == recipe['frames']
    root = data['Hips','translation']
    speed = max(math.dist(a,b)*30 for a,b in zip(root,root[1:]))
    joint_step = max(angle(a,b) for key,rows in data.items() if key[1]=='rotation' for a,b in zip(rows,rows[1:]))
    row = dict(file=path.name,frames=n,seconds=recipe['nativeSeconds'],rootPeakMetresPerSecond=speed,localPeakDegreesPerFrame=joint_step)
    if recipe.get('source'):
        source_doc, source = tracks(Path(recipe['source'])); assert source.keys() == data.keys()
        ranges = recipe.get('ranges',[[recipe['begin'],recipe['end']]])
        fixed = lambda f: not any(begin < f < end-1 for begin,end in ranges)
        worst = 0
        for key,values in source.items():
            for f,(a,b) in enumerate(zip(values,data[key])):
                if key not in body or fixed(f) or (recipe.get('preserveRoot') and key[0]=='Hips'):
                    error = delta(a,b,key[1]=='rotation'); worst = max(worst,error)
                    assert error < 2e-6,(path.name,key,f,error)
        for f in range(n):
            actual = world(doc,data,f)
            assert all(math.isfinite(x) for p,q in actual for x in (*p,*q))
            if fixed(f):
                expected = world(source_doc,source,f)
                assert max(math.dist(a[0],b[0]) for a,b in zip(actual,expected)) < 2e-5
        row['preservationMaxComponent'] = worst
    if recipe.get('command') == 'extend':
        _, source = tracks(Path(recipe['history'])); prefix = len(next(iter(source.values())))
        assert data.keys() == source.keys()
        for key,values in source.items():
            assert max(delta(a,b,key[1]=='rotation') for a,b in zip(values,data[key])) < 2e-6
            if key not in body:
                assert max(delta(values[-1],b,key[1]=='rotation') for b in data[key][prefix:]) < 2e-6
        row['preservedPrefixFrames'] = prefix
    # Full independent world FK of serialized targets, not native report reuse.
    # New SOMA-30 assets use the canonical parent order written by the exporter.
    if len(doc['nodes']) == 30 and not recipe.get('loop'):
        errors = []
        for target in recipe.get('constraintErrorAfterPostprocessing',[]):
            actual = world(doc,data,target['frame']); wanted = world(doc,data,target['frame'],target)
            if not target['anchored']:
                shift = [actual[0][0][k]-wanted[0][0][k] if k!=1 else 0 for k in range(3)]
                wanted = [(tuple(p[k]+shift[k] for k in range(3)),q) for p,q in wanted]
            indices = list(range(30)) if not target['effectors'] else [j for bit,j in enumerate((24,28,13,19)) if target['effectors']&(1<<bit)]
            position = max(math.dist(actual[j][0],wanted[j][0]) for j in indices)
            rotation = max(angle(actual[j][1],wanted[j][1]) for j in indices)
            assert abs(position-target['positionMaxMetres']) < 3e-5,(path.name,position,target)
            assert abs(rotation-target['rotationMaxDegrees']) < .15,(path.name,rotation,target['rotationMaxDegrees'])
            errors.append(dict(frame=target['frame'],effectors=target['effectors'],positionMaxMetres=position,rotationMaxDegrees=rotation))
        if errors: row['constraints'] = errors
    if recipe.get('loop'):
        assert max(delta(values[0],values[-1],key[1]=='rotation') for key,values in data.items()) < 2e-6
    if recipe.get('pathErrors'):
        for point in recipe['pathErrors']:
            p = root[point['frame']]; error = math.hypot(p[0]-point['x'],p[2]-point['z'])
            assert abs(error-point['metres']) < 1e-5
        for point in recipe.get('requestedPathErrors',[]):
            if 'headingDegrees' in point and len(doc['nodes'])==30:
                w = world(doc,data,point['frame']); right,left = w[26][0],w[22][0]
                heading = math.degrees(math.atan2(right[2]-left[2],left[0]-right[0]))
                error = math.remainder(heading-point['headingDegrees'],360)
                assert abs(math.remainder(error-point['headingErrorDegrees'],360)) < .001
        row['pathMaxMetres'] = max(p['metres'] for p in recipe['pathErrors'])
        row['requestedPathMaxMetres'] = max(p['metres'] for p in recipe['requestedPathErrors'])
    measurements.append(row)
print(json.dumps(measurements,indent=2))
