"""Development-only pinned DART graph compiler/reference. Never shipped or run by users.
Requires torch 2.7.1+cpu, onnx 1.18.0, onnxruntime 1.23.2; see dart-export-requirements.txt.
"""
import argparse,subprocess
import sys,types,importlib.util, pathlib, io,pickle,hashlib,json
from dart_graph_codegen import emit as emit_graph
import numpy as np
import torch,onnx,onnxruntime as ort
from types import SimpleNamespace as NS
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=pathlib.Path,required=True)
parser.add_argument('--clip-source',type=pathlib.Path,required=True)
parser.add_argument('--models',type=pathlib.Path,required=True)
parser.add_argument('--emit',type=pathlib.Path,required=True)
args=parser.parse_args()
if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!='bb67ae1ed6ce051080468bf15bc6e54a6c3f8417':raise ValueError('DART source revision mismatch')
subprocess.run(['git','-C',str(args.source),'diff','--quiet','HEAD','--'],check=True)
pins={'denoiser.pt':'e6c550641deaaccb6313f3529a4ca6602eb00d675dcff3aa47217825b5a43615','vae.pt':'b818cc8704ad9b408c745cac48943c99bf5402ffca521b25ffaf6639a5d5d276','policy.pt':'81af5cda56e519afadb7bdf74ba82322cc79f903c1cbea151b05fdc71f460a0e','clip.pt':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af','stats.pkl':'cb442539380d15a5763b799ba063d9b0b6574a751a03ae0c86d4a6f24bff5bd4'}
for name,digest in pins.items():
 if hashlib.sha256((args.models/name).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
for name,digest in {'model.py':'9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c','simple_tokenizer.py':'d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad','bpe_simple_vocab_16e6.txt.gz':'924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'}.items():
 if hashlib.sha256((args.clip_source/name).read_bytes()).hexdigest()!=digest:raise ValueError('CLIP source hash mismatch: '+name)
sys.path.insert(0,str(args.source.resolve()))

sys.modules['clip']=types.ModuleType('clip') # imported but unused by denoiser module
from model.mld_denoiser import DenoiserTransformer
from model.mld_vae import AutoMldVae
from control.policy.policy import PolicyReachLocationMLP
torch.set_num_threads(4); torch.manual_seed(1)
torch.backends.mha.set_fastpath_enabled(False)
base=args.models
emit=args.emit
clip_source=args.clip_source.resolve()
class Safe(pickle.Unpickler):
 def find_class(self,m,n):
  if (m,n)==('torch.storage','_load_from_bytes'):return lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
  if (m,n)==('torch._utils','_rebuild_tensor_v2'):return torch._utils._rebuild_tensor_v2
  if (m,n)==('collections','OrderedDict'):
   from collections import OrderedDict
   return OrderedDict
  raise ValueError((m,n))
mean,std=Safe(open(base/'stats.pkl','rb')).load()
np.savez(base/'stats.npz',mean=mean.numpy(),std=std.numpy())
def state(name):
 x=torch.load(base/(name+'.pt'),map_location='cpu',weights_only=True)
 return x.get('model_state_dict',x)
den=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,history_shape=(2,276),noise_shape=(1,256),cond_mask_prob=.1).eval()
den.load_state_dict(state('denoiser'))
vae=AutoMldVae(276,latent_dim=(1,256),h_dim=256,ff_size=1024,num_layers=7,num_heads=4).eval()
vae.load_state_dict(state('vae'))
vae.latent_std=state('vae')['latent_std'];vae.latent_mean=state('vae')['latent_mean']
structure={};offset=0
for k,n in [('goal_dir',3),('goal_dist',1),('goal_text_embedding',512),('motion',552),('scene',1)]:
 structure[k]={'numel':n,'start_idx':offset,'end_idx':offset+n};offset+=n
policy=PolicyReachLocationMLP(NS(observation_structure=structure,action_structure={'numel':256},latent_dim=512,activation='lrelu',pred_std=0,n_blocks=2,use_lora=0,lora_rank=16,use_zero_init=1,use_tanh_scale=1,min_log_std=-1,max_log_std=1)).eval()
policy.load_state_dict(state('policy'))
class Den(torch.nn.Module):
 def __init__(self):super().__init__();self.m=den
 def forward(self,noise,timestep,history,text):return self.m(noise,timestep,{'history_motion_normalized':history,'text_embedding':text})
class Dec(torch.nn.Module):
 def __init__(self):super().__init__();self.m=vae
 def forward(self,latent,history):return self.m.decode(latent,history,8,scale_latent=True)
class Policy(torch.nn.Module):
 def __init__(self):super().__init__();self.m=policy
 def forward(self,obs):
  z=self.m.actor(self.m.get_embedding(obs))
  return torch.cat((torch.tanh(z)*4,self.m.actor_logstd.clamp(-1,1).exp()),dim=1)
spec=importlib.util.spec_from_file_location('clip_model',clip_source/'model.py');cm=importlib.util.module_from_spec(spec);spec.loader.exec_module(cm)
jit=torch.jit.load(str(base/'clip.pt'),map_location='cpu')
clip=cm.build_model(jit.state_dict()).float().eval();del jit
class Text(torch.nn.Module):
 def __init__(self):super().__init__();self.m=clip
 def forward(self,tokens):return self.m.encode_text(tokens).float()
sys.path.insert(0,str(clip_source))
from simple_tokenizer import SimpleTokenizer
tok=SimpleTokenizer()
tokens=[]
for text in ['walk','run','hop on left leg']:
 ids=[49406]+tok.encode(text)+[49407];tokens.append(ids+[0]*(77-len(ids)))
(base/'tokens.json').write_text(json.dumps(tokens))
raw={name:(base/(name+'.pt')).read_bytes() for name in ['denoiser','vae','policy','clip']}
fragments=[]; entries=[];cursor=0;references=[]
for name,model,args,inputs,outputs in [
 ('denoiser',Den(),(torch.randn(1,1,256),torch.tensor([9]),torch.randn(1,2,276),torch.randn(1,512)),['noise','timestep','history','text'],['x0']),
 ('decoder',Dec(),(torch.randn(1,1,256),torch.randn(1,2,276)),['latent','history'],['future']),
 ('policy',Policy(),(torch.randn(1,1069),),['observation'],['distribution']),
 ('text',Text(),(torch.tensor([tokens[0]],dtype=torch.int64),),['tokens'],['embedding'])]:
 model.eval();path=base/(name+'.onnx')
 torch.onnx.export(model,args,str(path),opset_version=17,input_names=inputs,output_names=outputs,do_constant_folding=False,dynamo=False)
 sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
 with torch.no_grad():ref=model(*args).numpy()
 references.append({'name':name,'inputs':[v.numpy().reshape(-1).tolist() for v in args if v.dtype==torch.float32],'output':ref.reshape(-1).tolist()})
 actual=sess.run(None,{k:v.numpy() for k,v in zip(inputs,args)})[0]
 err=np.max(np.abs(ref-actual));print(name,'parity',err,flush=True)
 assert err<.002
 graph=onnx.load(path)
 for node in graph.graph.node:
  if node.op_type=='Constant':
   for a in node.attribute:
    if a.HasField('t') and len(a.t.raw_data)>128:
     values=onnx.numpy_helper.to_array(a.t)
     assert np.all((values==0)|np.isneginf(values)),('unexpected large constant',name,node.name)
 owner={'decoder':'vae','text':'clip'}.get(name,name)
 for t in graph.graph.initializer:
  data=t.raw_data
  pos=raw[owner].find(data);half=False
  if pos<0 and t.data_type==1:
   f=np.frombuffer(data,dtype='<f4');halfdata=f.astype('<f2').tobytes()
   if np.array_equal(np.frombuffer(halfdata,dtype='<f2').astype('<f4'),f):
    pos=raw[owner].find(halfdata);half=pos>=0
  if pos<0:raise ValueError(('initializer missing from source',name,t.name,len(data)))
  entries.append((owner,pos,len(data),cursor,half))
  t.ClearField('raw_data');t.data_location=onnx.TensorProto.EXTERNAL
  for key,value in [('location','dart-weights.bin'),('offset',str(cursor)),('length',str(len(data)))]:
   e=t.external_data.add();e.key=key;e.value=value
  cursor+=len(data)
 fragments.append(emit_graph(name,graph))
 (base/(name+'-external.onnx')).write_bytes(graph.SerializeToString())
 print(name,'graph metadata',len(graph.SerializeToString()),'initializers',len(graph.graph.initializer),flush=True)
out='// Generated operator/schema code; all learned tensor bytes remain in external pinned checkpoints.\n'+''.join(fragments)
out+='static constexpr DartSlice dart_slices[]={\n'+''.join('{"%s.pt",%d,%d,%d,%s},\n'%(n,p,l,o,'true' if h else 'false') for n,p,l,o,h in entries)+'};\n'
emit.write_text(out)
(base/'slices.json').write_text(json.dumps(entries))
(base/'reference.json').write_text(json.dumps(references))
print('total converted bytes',cursor)
