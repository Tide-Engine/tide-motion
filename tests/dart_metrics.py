"""Independent exported GLB/FK metrics. Development only; no runtime Python dependency."""
import json,math,sys
from pathlib import Path
from soma_metrics import positions,payload,metrics

def measure(path):
    recipe=json.loads(Path(str(path)+'.json').read_text())
    frames=positions(path);doc,binary=payload(path)
    assert len(frames)==recipe['frames'] and len(frames)<=18000
    parents={c:i for i,n in enumerate(doc['nodes']) for c in n.get('children',[])}
    rest={n['name']:n.get('translation',[0,0,0]) for n in doc['nodes']}
    max_length_error=0
    for f in frames:
        for child,parent in parents.items():
            a=f[doc['nodes'][child]['name']];b=f[doc['nodes'][parent]['name']]
            expected=math.sqrt(sum(v*v for v in rest[doc['nodes'][child]['name']]))
            max_length_error=max(max_length_error,abs(math.dist(a,b)-expected))
    assert max_length_error<2e-5,'animated bone lengths'
    segments=[]
    for segment in recipe['metrics']['segments']:
        first,last=segment['firstFrame'],segment['lastFrame'];root=frames[last]['Hips']
        row={'text':segment['text'],'first':first,'last':last,'chunks':segment['chunks']}
        distances=[math.dist(frames[i]['Hips'],frames[i-1]['Hips']) for i in range(first+1,last+1)]
        row['meanRootSpeedMetresPerSecond']=sum(distances)/max(1,len(distances))*30
        if 'target' in segment:
            goal=segment['target'];error=math.hypot(root[0]-goal[0],root[2]-goal[1])
            assert abs(error-segment['modelGoalErrorMetres'])<1e-4,'recipe/export goal mismatch'
            row.update(errorMetres=error,reached=segment['reached'])
        if 1<=first<len(frames)-1:
            a,b,c=[frames[i]['Hips'] for i in [first-1,first,first+1]]
            row['seamStepMetres']=math.dist(b,c)
            row['seamVelocityChangeMetresPerSecond']=math.sqrt(sum(((c[k]-b[k])-(b[k]-a[k]))**2 for k in range(3)))*30
        row['leftFootPeak']=max(f['LeftFoot'][1] for f in frames[first:last+1]);row['rightFootPeak']=max(f['RightFoot'][1] for f in frames[first:last+1])
        segments.append(row)
    def accessor(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']];offset=v.get('byteOffset',0)+a.get('byteOffset',0)
        import struct
        return [struct.unpack_from('<4f',binary,offset+16*i) for i in range(a['count'])]
    angular=0
    for ch in doc['animations'][0]['channels']:
        if ch['target']['path']!='rotation':continue
        q=accessor(doc['animations'][0]['samplers'][ch['sampler']]['output'])
        for a,b in zip(q,q[1:]):
            assert all(math.isfinite(x) for x in a+b)
            angular=max(angular,2*math.acos(min(1,abs(sum(a[k]*b[k] for k in range(4)))))*180/math.pi)
    for c in recipe['metrics'].get('optimization',{}).get('constraints',[]):
        if 'axes' not in c:continue  # Historical recipes lack the explicit mask.
        actual=frames[c['frame']][c['joint']]
        error=math.sqrt(sum((actual[k]-c['target'][k])**2 for k in range(3) if c['axes'][k]))
        assert abs(error-c['exportedErrorMetres'])<1e-4,'optimized constraint/export mismatch'
    return dict(metrics(path),maxBoneLengthErrorMetres=max_length_error,maxJointStepDegrees=angular,seconds=recipe['metrics']['seconds'],segments=segments)

if __name__=='__main__':
    rows=[measure(p) for p in sorted(Path(sys.argv[1]).glob('*.glb'))]
    print(json.dumps(rows,indent=2))
