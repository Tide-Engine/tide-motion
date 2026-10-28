"""Independent developer-only GLB/FK/contact observations; never inference.

Usage: python tools/dex2hoi/measure.py CLIP.glb HISTORY.json OUTPUT_STEM [MARKERS.json]
Requires the pinned developer NumPy environment; not used by installed inference.
Choose a private ignored output stem. No checkpoint or evaluation motion is needed.
"""
import argparse
import hashlib
import json
import math
import pathlib
import struct

import numpy as np


def quat_matrix(q):
    x,y,z,w=np.array(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def matrix_quat(m):
    # Eigenvector method independent of native quaternion/matrix implementation.
    r=m[:3,:3]
    k=np.array([[r[0,0]-r[1,1]-r[2,2],r[1,0]+r[0,1],r[2,0]+r[0,2],r[2,1]-r[1,2]],
                [r[1,0]+r[0,1],r[1,1]-r[0,0]-r[2,2],r[2,1]+r[1,2],r[0,2]-r[2,0]],
                [r[2,0]+r[0,2],r[2,1]+r[1,2],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
                [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],r[0,0]+r[1,1]+r[2,2]]])/3
    _,v=np.linalg.eigh(k);q=v[:,-1]
    return q if q[3]>=0 else -q


def trs(p=(0,0,0),q=(0,0,0,1),s=(1,1,1)):
    m=np.eye(4);m[:3,:3]=quat_matrix(q)@np.diag(s);m[:3,3]=p;return m


def angle(a,b):
    # Angle between orthonormal rotation matrices; no quaternion sign ambiguity.
    d=a[:3,:3].T@b[:3,:3]
    return math.degrees(math.acos(float(np.clip((np.trace(d)-1)/2,-1,1))))


class Glb:
    def __init__(self,path):
        self.path=pathlib.Path(path);b=self.path.read_bytes()
        magic,version,length=struct.unpack_from('<III',b)
        assert magic==0x46546c67 and version==2 and length==len(b)
        self.doc=None;self.bin=b'';offset=12
        while offset<len(b):
            n,t=struct.unpack_from('<II',b,offset);payload=b[offset+8:offset+8+n];offset+=8+n
            if t==0x4e4f534a:self.doc=json.loads(payload)
            elif t==0x004e4942:self.bin=payload
        self.nodes=self.doc['nodes'];self.byname={n.get('name',''):i for i,n in enumerate(self.nodes)}
        self.parents=[-1]*len(self.nodes)
        for i,n in enumerate(self.nodes):
            for child in n.get('children',[]):
                assert self.parents[child]<0;self.parents[child]=i
        self.channels=[];self.times=np.array([0.])
        if self.doc.get('animations'):
            animation=self.doc['animations'][0]
            for channel in animation['channels']:
                sampler=animation['samplers'][channel['sampler']]
                t=self.accessor(sampler['input']).reshape(-1)
                v=self.accessor(sampler['output'])
                assert sampler.get('interpolation','LINEAR') in ['LINEAR','STEP']
                self.channels.append((channel['target']['node'],channel['target']['path'],t,v,sampler.get('interpolation','LINEAR')))
                if len(t)>len(self.times):self.times=t

    def accessor(self,index):
        a=self.doc['accessors'][index];v=self.doc['bufferViews'][a['bufferView']]
        assert v.get('buffer',0)==0 and not a.get('sparse')
        dtype={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        count={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        offset=v.get('byteOffset',0)+a.get('byteOffset',0);size=np.dtype(dtype).itemsize
        stride=v.get('byteStride',count*size)
        return np.ndarray((a['count'],count),dtype=dtype,buffer=self.bin,offset=offset,strides=(stride,size)).copy()

    def evaluate(self,seconds):
        p=[n.get('translation',[0,0,0]) for n in self.nodes]
        q=[n.get('rotation',[0,0,0,1]) for n in self.nodes]
        s=[n.get('scale',[1,1,1]) for n in self.nodes]
        for node,kind,t,v,interp in self.channels if seconds is not None else []:
            j=int(np.searchsorted(t,seconds,side='right')-1);j=max(0,min(j,len(t)-1));k=min(j+1,len(t)-1)
            u=0 if k==j or interp=='STEP' else float(np.clip((seconds-t[j])/(t[k]-t[j]),0,1))
            a=v[j].astype(float);b=v[k].astype(float)
            if kind=='rotation':
                d=float(np.dot(a,b));b=b if d>=0 else -b;d=abs(d)
                if d>.9995:value=a*(1-u)+b*u;value/=np.linalg.norm(value)
                else:
                    theta=math.acos(min(1,d));value=(math.sin((1-u)*theta)*a+math.sin(u*theta)*b)/math.sin(theta)
                q[node]=value
            else:
                value=a*(1-u)+b*u
                if kind=='translation':p[node]=value
                elif kind=='scale':s[node]=value
                else:raise ValueError(kind)
        local=[np.array(n['matrix']).reshape(4,4).T if 'matrix' in n else trs(p[i],q[i],s[i]) for i,n in enumerate(self.nodes)]
        world=[None]*len(local)
        def visit(i):
            if world[i] is None:world[i]=local[i] if self.parents[i]<0 else visit(self.parents[i])@local[i]
            return world[i]
        for i in range(len(local)):visit(i)
        return np.array(local),np.array(world)

    def mesh(self,root=None):
        _,world=self.evaluate(0);vertices=[];triangles=[]
        rootmat=np.linalg.inv(world[root]) if root is not None else np.eye(4)
        for i,n in enumerate(self.nodes):
            ancestor=i
            while ancestor>=0 and ancestor!=root:ancestor=self.parents[ancestor]
            if root is not None and ancestor<0:continue
            if 'mesh' not in n:continue
            for prim in self.doc['meshes'][n['mesh']]['primitives']:
                assert prim.get('mode',4)==4
                xyz=self.accessor(prim['attributes']['POSITION']).astype(float)
                transform=rootmat@world[i];xyz=(np.c_[xyz,np.ones(len(xyz))]@transform.T)[:,:3]
                offset=len(vertices);vertices.extend(xyz.tolist())
                indices=self.accessor(prim['indices']).reshape(-1) if 'indices' in prim else np.arange(len(xyz))
                triangles.extend((indices.reshape(-1,3)+offset).tolist())
        return np.array(vertices,dtype=float),triangles


def stats(values):
    x=np.array(values,dtype=float)
    return {'min':float(x.min()),'mean':float(x.mean()),'p95':float(np.percentile(x,95)),'max':float(x.max())} if x.size else None


def motion_stats(transforms,prefix=10,fps=30):
    positions=transforms[:,:3,3];steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    rotations=[angle(transforms[i],transforms[i+1]) for i in range(len(transforms)-1)]
    return {'generatedPositionBounds':{'min':positions[prefix:].min(0).tolist(),'max':positions[prefix:].max(0).tolist()},
            'generatedTravelMetres':float(steps[prefix:].sum()),'generatedStepMetres':stats(steps[prefix:]),
            'generatedSpeedMetresPerSecond':stats(steps[prefix:]*fps),'generatedRotationStepDegrees':stats(rotations[prefix:]),
            'prefixToGeneratedPositionStepMetres':float(steps[prefix-1]),'prefixToGeneratedRotationStepDegrees':float(rotations[prefix-1])}


def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def surface_distances(points,vertices,triangles):
    """Exact point-to-triangle distance, including triangle edge fallback."""
    tri=vertices[np.array(triangles,dtype=int)];a=tri[:,0];b=tri[:,1];c=tri[:,2]
    ab=b-a;ac=c-a;normal=np.cross(ab,ac);norm2=(normal*normal).sum(1)
    valid=norm2>1e-20;result=[]
    for point in points:
        distance=np.full(len(tri),np.inf)
        signed=((point-a)*normal).sum(1);projected=point-(signed/np.maximum(norm2,1e-30))[:,None]*normal
        ap=projected-a;d00=(ab*ab).sum(1);d01=(ab*ac).sum(1);d11=(ac*ac).sum(1);d20=(ap*ab).sum(1);d21=(ap*ac).sum(1)
        denom=np.maximum(d00*d11-d01*d01,1e-30);v=(d11*d20-d01*d21)/denom;w=(d00*d21-d01*d20)/denom
        inside=valid&(v>=-1e-10)&(w>=-1e-10)&(v+w<=1+1e-10)
        distance[inside]=(signed[inside]**2/norm2[inside])
        for start,end in [(a,b),(b,c),(c,a)]:
            edge=end-start;t=np.clip(((point-start)*edge).sum(1)/np.maximum((edge*edge).sum(1),1e-30),0,1)
            closest=start+t[:,None]*edge;distance=np.minimum(distance,((point-closest)**2).sum(1))
        result.append(math.sqrt(max(0,float(distance.min()))))
    return np.array(result)


def analyze(clip_path,history_path,stem,markers_path=None):
    clip=Glb(clip_path);hpath=pathlib.Path(history_path);spec=json.loads(hpath.read_text())
    history=Glb(hpath.parent/spec['history']);names=[n['name'] for n in history.nodes];body_indices=[clip.byname[n] for n in names]
    samples=[clip.evaluate(float(t)) for t in clip.times];local=np.array([x[0] for x in samples]);world=np.array([x[1] for x in samples]);prefix=10;fps=30
    assert len(samples)==70 and np.max(np.abs(np.diff(clip.times)-1/fps))<1e-6
    source=[history.evaluate(float(t)) for t in history.times]
    prefix_positions=[];prefix_angles=[]
    for f in range(10):
        for j,name in enumerate(names):
            target=local[f,clip.byname[name]];original=source[f][0][j]
            prefix_positions.append(np.linalg.norm(target[:3,3]-original[:3,3]));prefix_angles.append(angle(target,original))
    bones=spec['bones'];wrists={'left':clip.byname[bones[10]],'right':clip.byname[bones[34]]};root=clip.byname[bones[0]]
    metrics={'version':1,'clip':str(pathlib.Path(clip_path).resolve()),'clipSha256':sha(clip_path),'historyManifestSha256':sha(hpath),
             'frames':len(samples),'fps':fps,'prefixFrames':prefix,'generatedFrames':len(samples)-prefix,
             'scope':'Independent sampled GLB/FK observations. No thresholds here certify contact, anatomy or prompt adherence.',
             'finite':bool(np.isfinite(local).all() and np.isfinite(world).all()),
             'bodyPrefixPositionErrorMetres':stats(prefix_positions),'bodyPrefixRotationErrorDegrees':stats(prefix_angles),
             'root':motion_stats(world[:,root]),'hands':{k:motion_stats(world[:,j]) for k,j in wrists.items()},'objects':[]}
    recipe_path=pathlib.Path(str(clip_path)+'.json')
    if recipe_path.exists():
        recipe=json.loads(recipe_path.read_text());metrics['recipeSha256']=sha(recipe_path)
        metrics['generation']={key:recipe[key] for key in ['prompt','seed','samplingTimesteps','guidance','bundleSha256','upstreamRevision','device','threads','onnxRuntime','inferenceSeconds'] if key in recipe}
    provenance_path=hpath.parent/'provenance.json'
    if provenance_path.exists():metrics['inputProvenance']=json.loads(provenance_path.read_text());metrics['inputProvenanceSha256']=sha(provenance_path)
    lengths=[]
    for name in names:
        j=clip.byname[name];parent=clip.parents[j]
        if parent in body_indices:
            length=np.linalg.norm(world[:,j,:3,3]-world[:,parent,:3,3],axis=1);lengths.append(float(np.ptp(length)))
    metrics['boneLengthPeakVariationMetres']=max(lengths,default=0)
    finger_indices=list(range(11,31))+list(range(35,55));finger_changes=[]
    for index in finger_indices:
        j=clip.byname[bones[index]];finger_changes.append(max(angle(local[prefix-1,j],local[f,j]) for f in range(prefix,len(samples))))
    metrics['mappedFingerLocalChangeFromPrefixDegrees']=stats(finger_changes)
    markers=[]
    if markers_path:
        marker_doc=json.loads(pathlib.Path(markers_path).read_text());markers=marker_doc.get('grips',marker_doc.get('markers',marker_doc if isinstance(marker_doc,list) else []))
    head=clip.byname[bones[5]];head_index=names.index(bones[5]);root_index=names.index(bones[0]);rest_world=history.evaluate(None)[1]
    def view_rotations(node,history_index):
        result=[]
        for f in range(len(samples)):
            m=np.eye(4);m[:3,:3]=world[f,node,:3,:3]@rest_world[history_index,:3,:3].T;result.append(matrix_quat(m))
        return np.round(result,8).tolist()
    viewer={'version':1,'fps':fps,'prefixFrames':prefix,'sourceClip':pathlib.Path(clip_path).name,
            'prompt':metrics.get('generation',{}).get('prompt',''),'seed':metrics.get('generation',{}).get('seed'),
            'head':head_index,'root':root_index,
            'headViewRotations':view_rotations(head,head_index),'rootViewRotations':view_rotations(root,root_index),
            'joints':[{'name':name,'parent':names.index(history.nodes[history.parents[j]]['name']) if history.parents[j]>=0 else -1} for j,name in enumerate(names)],
            'frames':np.round(world[:,body_indices][:,:,:3,3],7).tolist(),'objects':[]}
    for slot,object_spec in enumerate(spec['objects']):
        if not object_spec.get('present',True) or not object_spec.get('exportTrack',True):continue
        name=object_spec['name'];j=clip.byname[name];poses=world[:,j];source_mesh=Glb(hpath.parent/object_spec['mesh']) if object_spec.get('mesh') else None
        vertices,triangles=clip.mesh(j)
        cloud=np.array(object_spec['points'],dtype=float)
        ob={'name':name,'slot':slot,'motion':motion_stats(poses),'prefixPositionErrorMetres':stats([np.linalg.norm(poses[f,:3,3]-object_spec['poses'][f]['position']) for f in range(prefix)]),
            'prefixRotationErrorDegrees':stats([angle(poses[f],trs(q=object_spec['poses'][f]['rotation'])) for f in range(prefix)]),
            'geometry':{'meshBoundsMin':vertices.min(0).tolist(),'meshBoundsMax':vertices.max(0).tolist(),'cloudBoundsMin':cloud.min(0).tolist(),'cloudBoundsMax':cloud.max(0).tolist(),
                        'meshDimensions':np.ptp(vertices,axis=0).tolist(),'cloudDimensions':np.ptp(cloud,axis=0).tolist()}}
        ob['geometry']['cloudToActualMeshSurfaceMetres']=stats(surface_distances(cloud,vertices,triangles))
        marker=next((m for m in markers if m.get('objectName',m.get('object'))==name),None)
        if marker:
            hand=clip.byname[marker.get('handBone',marker.get('hand'))]
            hm=marker.get('handMarker',{'position':[0,0,0],'rotation':[0,0,0,1]});om=marker.get('objectMarker',{'position':[0,0,0],'rotation':[0,0,0,1]})
            hand_marker=trs(hm.get('position',[0,0,0]),hm.get('rotation',[0,0,0,1]));object_marker=trs(om.get('position',[0,0,0]),om.get('rotation',[0,0,0,1]));mode='explicit authored markers'
        else:
            hand=wrists['right' if slot==0 else 'left'];hand_marker=np.eye(4)
            # Fallback defines a stable reference grip from the user's last prefix
            # only. Its zero initial gap is a calibration, not proven grasp.
            object_marker=np.linalg.inv(poses[prefix-1])@world[prefix-1,hand];mode='calibrated from frame9, not independently authored contact'
        hp=world[:,hand]@hand_marker;op=poses@object_marker;relative=np.linalg.inv(hp)@op
        gaps=np.linalg.norm(hp[:,:3,3]-op[:,:3,3],axis=1);orient=[angle(hp[f],op[f]) for f in range(len(samples))]
        drift=[np.linalg.norm(relative[f,:3,3]-relative[prefix-1,:3,3]) for f in range(prefix,len(samples))]
        orientation_drift=[angle(relative[prefix-1],relative[f]) for f in range(prefix,len(samples))]
        ob['grip']={'measurement':mode,'hand':clip.nodes[hand]['name'],'generatedPositionGapMetres':stats(gaps[prefix:]),'generatedOrientationGapDegrees':stats(orient[prefix:]),
                    'generatedRelativeDriftFromPrefixMetres':stats(drift),'generatedRelativeOrientationDriftFromPrefixDegrees':stats(orientation_drift),
                    'prefixToGeneratedGapMetres':float(gaps[prefix]),'gapByFrameMetres':np.round(gaps,7).tolist()}
        tips=[38,42,46,50,54] if hand==wrists['right'] else [14,18,22,26,30]
        finger_indices=[clip.byname[bones[index]] for index in tips];finger_distances=[]
        for f in range(prefix,len(samples)):
            local_tips=(np.linalg.inv(poses[f])@np.c_[world[f,finger_indices,:3,3],np.ones(5)].T).T[:,:3]
            finger_distances.append(surface_distances(local_tips,vertices,triangles))
        ob['grip']['generatedFingertipBoneToMeshSurfaceMetres']=stats(np.array(finger_distances).reshape(-1))
        ob['grip']['generatedNearestFingertipBoneToMeshSurfaceMetres']=stats(np.array(finger_distances).min(axis=1))
        ob['grip']['fingertipMetricScope']='Bone endpoints vs any rigid mesh triangle (including blade), not designated handle contact. Not skin contact, penetration or grasp certification.'
        metrics['objects'].append(ob)
        viewer['objects'].append({'name':name,'vertices':np.round(vertices,7).tolist(),'triangles':triangles,
            'positions':np.round(poses[:,:3,3],7).tolist(),'rotations':np.round(np.array([matrix_quat(p) for p in poses]),8).tolist(),
            'hand':names.index(clip.nodes[hand]['name']),'handMarker':{'position':hand_marker[:3,3].tolist(),'rotation':matrix_quat(hand_marker).tolist()},
            'objectMarker':{'position':object_marker[:3,3].tolist(),'rotation':matrix_quat(object_marker).tolist()},
            'markerPositions':np.round(op[:,:3,3],7).tolist(),'handMarkerPositions':np.round(hp[:,:3,3],7).tolist(),
            'gapByFrameMetres':gaps.tolist()})
    out=pathlib.Path(stem);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    out.with_suffix('.motion.json').write_text(json.dumps(viewer,separators=(',',':'))+'\n')
    print(json.dumps({'clip':str(clip_path),'objects':[{ 'name':x['name'],'gapMeanMetres':x['grip']['generatedPositionGapMetres']['mean'],'gapMaxMetres':x['grip']['generatedPositionGapMetres']['max'],'seamStepMetres':x['motion']['prefixToGeneratedPositionStepMetres']} for x in metrics['objects']]},indent=2))
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('clip');p.add_argument('history');p.add_argument('output_stem');p.add_argument('markers',nargs='?');a=p.parse_args()
    analyze(a.clip,a.history,a.output_stem,a.markers)
