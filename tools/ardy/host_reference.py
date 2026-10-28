"""Generate own-prompt fixed-noise host fixtures using the pinned upstream sampler.
Developer-only. No dataset or evaluation motions are used.
"""
import argparse
import json
import pathlib
import sys
import struct
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

p=argparse.ArgumentParser()
p.add_argument('--source',type=pathlib.Path,required=True)
p.add_argument('--checkpoint',type=pathlib.Path,required=True)
p.add_argument('--text',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--history-glb',type=pathlib.Path)
a=p.parse_args()
sys.path.insert(0,str(a.source.resolve()))
torch.set_num_threads(8)
torch.backends.mha.set_fastpath_enabled(False)
conf=OmegaConf.load(a.checkpoint/'config.yaml')
cfg=OmegaConf.to_container(OmegaConf.merge(conf,{'checkpoint_dir':str(a.checkpoint.resolve())}),resolve=True)
cfg.pop('checkpoint_dir')
model=instantiate(cfg,text_encoder=None,device='cpu')
model.eval()
t=json.loads(a.text.read_text())
text=torch.tensor(t['text']).reshape(1,1,4096)
mask=torch.zeros(1,183,369)
observed=torch.zeros_like(mask)
mask[:,150,0]=mask[:,150,2]=1
observed[:,150,0]=2
observed[:,150,2]=1
observed=model.motion_rep.normalize(observed)*mask
noises=[]
original=torch.randn
def capture(*args,**kwargs):
    v=original(*args,**kwargs)
    noises.extend(v.flatten().tolist())
    return v
torch.manual_seed(431)
torch.randn=capture
with torch.no_grad():
    output=model(texts=[t['prompt']],num_frames=183,num_denoising_steps=10,
                 pad_mask=torch.ones(1,183,dtype=torch.bool),first_heading_angle=torch.zeros(1),
                 motion_mask=mask,observed_motion=observed,cfg_weight=(2.,2.),
                 text_feat=text,text_pad_mask=torch.ones(1,1,dtype=torch.bool),
                 crop_history_length=240,progress_bar=lambda x:x)
torch.randn=original
raw=model.motion_rep.unnormalize(output)
a.output.write_text(json.dumps({'prompt':t['prompt'],'text':t['text'],'noise':noises,'features':raw.flatten().tolist()}))
print('wrote',a.output,'noise',len(noises),'frames',raw.shape[1])
if a.history_glb:
    b=a.history_glb.read_bytes()
    jlen=struct.unpack_from('<I',b,12)[0]
    gltf=json.loads(b[20:20+jlen])
    data=b[28+jlen:]
    def accessor(index):
        acc=gltf['accessors'][index];view=gltf['bufferViews'][acc['bufferView']]
        width={'SCALAR':1,'VEC3':3,'VEC4':4}[acc['type']]
        return np.frombuffer(data,dtype='<f4',count=acc['count']*width,
                             offset=view.get('byteOffset',0)+acc.get('byteOffset',0)).reshape(acc['count'],width).copy()
    rotations=np.zeros((64,30,4),dtype=np.float32);rotations[:,:,3]=1
    positions=np.zeros((64,3),dtype=np.float32)
    anim=gltf['animations'][0]
    for ch in anim['channels']:
        node=ch['target']['node'];name=gltf['nodes'][node].get('name')
        if name not in model.skeleton.bone_order_names:continue
        joint=model.skeleton.bone_order_names.index(name)
        values=accessor(anim['samplers'][ch['sampler']]['output'])[:64]
        if ch['target']['path']=='rotation':rotations[:,joint]=values
        elif ch['target']['path']=='translation' and joint==0:positions[:]=values
    from ardy.geometry import quaternion_to_matrix
    matrices=quaternion_to_matrix(torch.tensor(rotations[:,:,[3,0,1,2]])).unsqueeze(0)
    features=model.motion_rep(matrices,torch.tensor(positions).unsqueeze(0),to_normalize=False)
    history_path=a.output.parent/'history.json'
    history_path.write_text(json.dumps({'glb':str(a.history_glb.resolve()),'features':features.flatten().tolist(),
                                       'parents':model.skeleton.joint_parents.tolist(),
                                       'names':model.skeleton.bone_order_names,
                                       'neutral':model.skeleton.neutral_joints.tolist()}))
    print('wrote',history_path,flush=True)
    noises=[]
    torch.manual_seed(432)
    torch.randn=capture
    with torch.no_grad():
        imported=model(texts=[t['prompt']],num_frames=185,num_denoising_steps=10,
                       pad_mask=torch.ones(1,185,dtype=torch.bool),first_heading_angle=None,
                       motion_mask=None,observed_motion=None,cfg_weight=(2.,2.),
                       text_feat=text,text_pad_mask=torch.ones(1,1,dtype=torch.bool),
                       init_history_sequence=model.motion_rep.normalize(features),
                       progress_bar=lambda x:x)
    torch.randn=original
    imported=model.motion_rep.unnormalize(imported)[:,64:185]
    (a.output.parent/'imported.json').write_text(json.dumps({'prompt':t['prompt'],'text':t['text'],
        'noise':noises,'features':imported.flatten().tolist(),'historyFeatures':features.flatten().tolist(),'glb':str(a.history_glb.resolve())}))
