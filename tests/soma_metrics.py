"""Small authored-prompt GLB comparison; standard library, development only.
Foot slide is a geometric proxy, not the upstream benchmark/contact metric.
"""
import json
import math
from pathlib import Path
import struct
import sys


def payload(path):
    raw = path.read_bytes()
    size = struct.unpack_from('<I', raw, 12)[0]
    return json.loads(raw[20:20+size]), raw[28+size:]


def positions(path):
    doc, binary = payload(path)
    nodes = doc['nodes']
    channels = {}
    def accessor(index):
        a = doc['accessors'][index]; v = doc['bufferViews'][a['bufferView']]
        width = {'SCALAR':1,'VEC3':3,'VEC4':4}[a['type']]
        offset = v.get('byteOffset',0)+a.get('byteOffset',0)
        return [struct.unpack_from('<'+'f'*width,binary,offset+i*width*4) for i in range(a['count'])]
    anim = doc['animations'][0]
    for c in anim['channels']:
        channels[c['target']['node'],c['target']['path']] = accessor(anim['samplers'][c['sampler']]['output'])
    frames = len(next(iter(channels.values())))
    parents = {c:i for i,n in enumerate(nodes) for c in n.get('children',[])}
    def mul(a,b):
        x,y,z,w=a; X,Y,Z,W=b
        return (w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z)
    result=[]
    for f in range(frames):
        world={}
        def visit(i):
            if i in world:return world[i]
            p=channels.get((i,'translation'),[nodes[i].get('translation',[0,0,0])]*frames)[f]
            q=channels.get((i,'rotation'),[nodes[i].get('rotation',[0,0,0,1])]*frames)[f]
            if i in parents:
                pp,pq=visit(parents[i]);r=mul(mul(pq,(*p,0)),(-pq[0],-pq[1],-pq[2],pq[3]))
                p=tuple(pp[k]+r[k] for k in range(3));q=mul(pq,q)
            world[i]=(p,q);return p,q
        result.append({n['name']:visit(i)[0] for i,n in enumerate(nodes) if 'name' in n})
    return result


def metrics(path):
    frames=positions(path)
    feet=['LeftFoot','RightFoot','LeftToeBase','RightToeBase']
    slide=[]
    for a,b in zip(frames,frames[1:]):
        for foot in feet:
            if max(a[foot][1],b[foot][1]) < .03:
                slide.append(math.hypot(a[foot][0]-b[foot][0],a[foot][2]-b[foot][2])*100)
    root=[f['Hips'] for f in frames]
    travel=math.hypot(root[-1][0]-root[0][0],root[-1][2]-root[0][2])
    return {'file':path.name,'frames':len(frames),'footPeakMetres':max(f[x][1] for f in frames for x in feet[:2]),
            'rootTravelMetres':travel,'slideCmPerFrame':sum(slide)/len(slide) if slide else None,'contactPairs':len(slide)}


if __name__=='__main__':
    folder=Path(sys.argv[1]); rows=[metrics(p) for p in sorted(folder.glob('*.glb'))]
    print(json.dumps(rows,indent=2))
    a=payload(folder/'v1.1-idle-1.glb')[1];b=payload(folder/'v1.1-repeat.glb')[1]
    assert a==b,'v1.1 motion changed after switching model'
