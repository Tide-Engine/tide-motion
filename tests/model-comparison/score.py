"""Independent GLB geometry proxies and paired summaries; development-only.
Thresholds are declared before generation results are inspected. These are not
semantic benchmark metrics or an automatic artistic-quality judgement.
"""
import json
import math
from pathlib import Path
import random
import statistics as st
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from soma_metrics import payload,positions


def quantile(values,p):
    a=sorted(values)
    return a[round((len(a)-1)*p)] if a else None


def angular_steps(path):
    doc,binary=payload(path);steps=[]
    for channel in doc['animations'][0]['channels']:
        if channel['target']['path']!='rotation':continue
        a=doc['accessors'][doc['animations'][0]['samplers'][channel['sampler']]['output']]
        v=doc['bufferViews'][a['bufferView']];offset=v.get('byteOffset',0)+a.get('byteOffset',0)
        qs=[struct.unpack_from('<ffff',binary,offset+i*16) for i in range(a['count'])]
        for x,y in zip(qs,qs[1:]):
            dot=sum(a*b for a,b in zip(x,y))/math.sqrt(sum(a*a for a in x)*sum(b*b for b in y))
            steps.append(math.degrees(2*math.acos(min(1,abs(dot)))))
    return steps


def score(path,item):
    frames=positions(path); n=len(frames)
    assert n==round(item['duration']*30),(path,n)
    assert all(math.isfinite(c) for f in frames for p in f.values() for c in p)
    feet=['LeftFoot','LeftToeBase','RightFoot','RightToeBase'];slide=[];possible=(n-1)*4
    for a,b in zip(frames,frames[1:]):
        for foot in feet:
            if max(a[foot][1],b[foot][1])<.03:
                slide.append(math.hypot(a[foot][0]-b[foot][0],a[foot][2]-b[foot][2])*100)
    root=[f['Hips'] for f in frames]
    travel=math.hypot(root[-1][0]-root[0][0],root[-1][2]-root[0][2])
    root_speed=[math.hypot(b[0]-a[0],b[2]-a[2])*30 for a,b in zip(root,root[1:])]
    peaks={side:max(f[side+'Foot'][1] for f in frames) for side in ['Left','Right']}
    hands={side:max(f[side+'Hand'][1]-f['Head'][1] for f in frames) for side in ['Left','Right']}
    airborne=[min(f[x][1] for x in feet)>.08 for f in frames]
    # Average hip heading uses the horizontal left-to-right hip line. Facing +Z
    # has left hip at +X; rotate its axis by 90 degrees in XZ.
    heading=math.atan2(-(frames[0]['LeftLeg'][2]-frames[0]['RightLeg'][2]),frames[0]['LeftLeg'][0]-frames[0]['RightLeg'][0])
    dx=root[-1][0]-root[0][0];dz=root[-1][2]-root[0][2]
    forward=dx*math.sin(heading)+dz*math.cos(heading)
    last=frames[-1];end_heading=math.atan2(-(last['LeftLeg'][2]-last['RightLeg'][2]),last['LeftLeg'][0]-last['RightLeg'][0])
    turn=abs(math.degrees(math.atan2(math.sin(end_heading-heading),math.cos(end_heading-heading))))
    drop=max(f['Head'][1] for f in frames)-min(f['Head'][1] for f in frames)
    case=item['case'];success=None
    if case=='walk':success=forward>.5
    elif case=='run':success=forward>.75 and sum(airborne)>=3
    elif case=='backward':success=forward<-.5
    elif case in ('rightkick','swordkick','posekick'):success=peaks['Right']>.45 and peaks['Right']>peaks['Left']+.15
    elif case=='leftkick':success=peaks['Left']>.45 and peaks['Left']>peaks['Right']+.15
    elif case=='jump':success=sum(airborne)>=3 and max(p[1] for p in root)-min(p[1] for p in root)>.12 and not any(airborne[-5:])
    elif case=='crouch':success=drop>.4 and st.mean(f['Head'][1] for f in frames[-5:])<st.mean(f['Head'][1] for f in frames[:5])-.3
    elif case=='wave':success=hands['Right']>0 and hands['Right']>hands['Left']+.1
    elif case=='turn':success=turn>135
    elif case=='walkwave':success=math.hypot(root[44][0]-root[0][0],root[44][2]-root[0][2])>.3 and max(f['RightHand'][1]-f['Head'][1] for f in frames[45:])>0
    elif case=='kicksequence':success=max(f['RightFoot'][1] for f in frames[:45])>.45 and max(f['LeftFoot'][1] for f in frames[45:])>.45
    angles=angular_steps(path)
    recipe=json.loads(path.with_suffix('.glb.json').read_text())
    assert recipe['model']=='nvidia/Kimodo-SOMA-RP-'+item['version']
    errors=recipe.get('constraints',[])
    pose_errors=[x['modelError']['positionMeanMetres']*100 for x in errors if 'modelError' in x]
    path_errors=[x['error']['hipsGroundMetres']*100 for x in errors if x['kind']=='path']
    return dict(item,success=success,slide=st.mean(slide) if slide else None,contactFraction=len(slide)/possible,
                penetrationPercent=sum(any(f[x][1]<-.02 for x in feet) for f in frames)/n*100,
                angularFlag60=max(angles)>60,maxJointStepDegrees=max(angles),jointStepP95Degrees=quantile(angles,.95),rootPeakSpeed=max(root_speed),
                travel=travel,forward=forward,rightFootPeak=peaks['Right'],leftFootPeak=peaks['Left'],headRange=drop,
                turnDegrees=turn,poseErrorCm=st.mean(pose_errors) if pose_errors else None,pathErrorCm=st.mean(path_errors) if path_errors else None)


def interval(deltas):
    # Paired cluster bootstrap; resample seed means across durations, never frames.
    rng=random.Random(1729);means=[]
    for _ in range(2000):means.append(st.mean(rng.choices(deltas,k=len(deltas))))
    return [quantile(means,.025),quantile(means,.975)]


def summarize(rows):
    out={}
    for case in sorted({r['case'] for r in rows}):
        pair={}
        for r in rows:
            if r['case']==case:pair.setdefault((r['seed'],r['duration']),{})[r['version']]=r
        assert all(set(p)=={'v1.1','v1.2'} for p in pair.values())
        result={'pairs':len(pair)}
        for version in ['v1.1','v1.2']:
            group=[p[version] for p in pair.values()]
            result[version]={'successes':sum(r['success'] for r in group) if group[0]['success'] is not None else None}
            for metric in ['slide','contactFraction','penetrationPercent','maxJointStepDegrees','jointStepP95Degrees','rootPeakSpeed','poseErrorCm','pathErrorCm']:
                a=[r[metric] for r in group if r[metric] is not None]
                result[version][metric]=st.mean(a) if a else None
        both=[p for p in pair.values() if all(p[v]['success'] is True for v in p)]
        result['bothSuccessPairs']=len(both)
        result['bothSuccessSlide']={v:st.mean(p[v]['slide'] for p in both if p[v]['slide'] is not None) if any(p[v]['slide'] is not None for p in both) else None for v in ['v1.1','v1.2']}
        result['v11OnlySuccess']=sum(p['v1.1']['success'] is True and p['v1.2']['success'] is False for p in pair.values())
        result['v12OnlySuccess']=sum(p['v1.2']['success'] is True and p['v1.1']['success'] is False for p in pair.values())
        result['paired']={} 
        for metric in ['slide','penetrationPercent','maxJointStepDegrees','poseErrorCm','pathErrorCm']:
            a=[p['v1.2'][metric]-p['v1.1'][metric] for p in pair.values() if all(p[v][metric] is not None for v in p)]
            if a:
                cluster=[]
                for seed in sorted({k[0] for k in pair}):
                    ds=[p['v1.2'][metric]-p['v1.1'][metric] for (s,d),p in pair.items() if s==seed and all(p[v][metric] is not None for v in p)]
                    if ds:cluster.append(st.mean(ds))
                result['paired'][metric]={'meanDeltaV12MinusV11':st.mean(a),'seedClusterBootstrap95':interval(cluster)}
        out[case]=result
    return out


if __name__=='__main__':
    work=Path(sys.argv[1]);protocol=json.loads((work/'protocol.json').read_text());rows=[]
    for i,item in enumerate(protocol['items']):
        rows.append(score(work/item['file'],item))
        if (i+1)%50==0:print(f'Scored {i+1}',file=sys.stderr)
    (work/'scores.json').write_text(json.dumps(rows,indent=2))
    (work/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
