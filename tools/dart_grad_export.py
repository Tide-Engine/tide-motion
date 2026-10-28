"""Compile native DART vector-Jacobian graphs. Development-only, pinned export environment.
Uses the same CLI arguments/raw pins as dart_export.py. No model tensors are emitted to Git.
"""
import sys,runpy,pathlib,json,hashlib
import numpy as np
import torch,onnx,onnxruntime as ort
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import core_aten_decompositions
from torch.onnx import register_custom_op_symbolic,symbolic_helper as sh
from dart_graph_codegen import emit
assert torch.__version__=="2.7.1+cpu" and torch.version.git_version=="e2d141dbde55c2a4370fac5165b0561b6af4798b", "Use the pinned gradient compiler"
def softmax(g,x,dim,half):return g.op('Softmax',x,axis_i=sh._parse_arg(dim,'i'))
def softmax_grad(g,grad,out,dim,dtype):
 axes=g.op('Constant',value_t=torch.tensor([sh._parse_arg(dim,'i')]))
 return g.op('Mul',out,g.op('Sub',grad,g.op('ReduceSum',g.op('Mul',grad,out),axes,keepdims_i=1)))
def scatter(g,x,src,dim,start,end,step):
 axis=sh._parse_arg(dim,'i');begin=sh._parse_arg(start,'i');stop=sh._parse_arg(end,'i');assert sh._parse_arg(step,'i')==1
 size=sh._get_tensor_dim_size(x,axis);assert size is not None
 begin=max(0,min(size,begin+size if begin<0 else begin));stop=max(0,min(size,stop+size if stop<0 else stop))
 return g.op('Concat',sh._slice_helper(g,x,axes=[axis],starts=[0],ends=[begin]),src,sh._slice_helper(g,x,axes=[axis],starts=[stop],ends=[size]),axis_i=axis)
register_custom_op_symbolic('aten::_softmax',softmax,17)
register_custom_op_symbolic('aten::_softmax_backward_data',softmax_grad,17)
register_custom_op_symbolic('aten::slice_scatter',scatter,17)
arguments=sys.argv.copy();output=pathlib.Path(arguments[arguments.index('--emit')+1]);sys.argv[sys.argv.index('--emit')+1]=str(output.parent/'dart-gradient-fixed-reference.inc')
g=runpy.run_path(str(pathlib.Path(__file__).with_name('dart_export.py')));base=g['base'];raw=g['raw'];entries=g['entries']
den=g['den'].eval().requires_grad_(False);vae=g['vae'].eval().requires_grad_(False)
def decode(z,h):return vae.decode(z.permute(1,0,2),h,8,scale_latent=True)
def decode_grad(z,h,v):
 _,backward=torch.func.vjp(decode,z,h);a,b=backward(v);return torch.cat((a.flatten(1),b.flatten(1)),dim=1)
def denoise_grad(x,t,h,text,v):
 _,backward=torch.func.vjp(lambda a,b:den(a,t,{'history_motion_normalized':b,'text_embedding':text}),x,h)
 a,b=backward(v);return torch.cat((a.flatten(1),b.flatten(1)),dim=1)
def r6(v):
 a,b=v[...,:3],v[...,3:];x=torch.nn.functional.normalize(a,dim=-1);y=torch.nn.functional.normalize(b-(x*b).sum(-1,keepdim=True)*x,dim=-1);z=torch.linalg.cross(x,y)
 return torch.stack((x,y,z),dim=-2)
def transition(future,world,delta):
 h=future[:,6:8];R=world[:,:9].reshape(1,3,3);T=world[:,9:].reshape(1,1,3)
 origin=h[:,0,144:147];x=h[:,0,150:153]-h[:,0,147:150];x=torch.cat((x[:,:2],torch.zeros_like(x[:,:1])),dim=-1);x=torch.nn.functional.normalize(x,dim=-1)
 z=torch.cat((torch.zeros_like(x[:,:2]),torch.ones_like(x[:,:1])),dim=-1);Q=torch.stack((x,torch.linalg.cross(z,x),z),dim=-1)
 transl=torch.einsum('bti,bij->btj',h[:,:,:3]+delta[:,None,:]-origin[:,None,:],Q)-delta[:,None,:]
 root=torch.einsum('bij,btjk->btik',Q.transpose(-1,-2),r6(h[:,:,3:9]))[:,:,:2,:].flatten(2)
 velocity=torch.einsum('bti,bij->btj',h[:,:,135:138],Q)
 rotation=torch.einsum('bij,btjk,bkl->btil',Q.transpose(-1,-2),r6(h[:,:,138:144]),Q)[:,:,:2,:].flatten(2)
 joints=torch.einsum('btki,bij->btkj',h[:,:,144:210].reshape(1,2,22,3)-origin[:,None,None,:],Q).flatten(2)
 joint_delta=torch.einsum('btki,bij->btkj',h[:,:,210:].reshape(1,2,22,3),Q).flatten(2)
 history=torch.cat((transl,root,h[:,:,9:135],velocity,rotation,joints,joint_delta),dim=-1)
 newR=R@Q;newT=(R@origin[:,:,None]).squeeze(-1)+T[:,0]
 points=torch.einsum('bij,btkj->btki',R,future[:,:,144:210].reshape(1,8,22,3))+T[:,:,None,:]
 points=torch.stack((-points[...,0],points[...,2],points[...,1]),dim=-1)
 return torch.cat((history.flatten(1),newR.flatten(1),newT,points.flatten(1)),dim=1)
def transition_grad(future,world,delta,v):
 _,backward=torch.func.vjp(lambda f,w:transition(f,w,delta),future,world);a,b=backward(v);return torch.cat((a.flatten(1),b),dim=1)
model_parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
soma_parents=[-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28]
soma_slots=[0,3,6,9,12,-1,15,-1,-1,-1,13,16,18,20,-1,-1,14,17,19,21,-1,-1,1,4,7,10,2,5,8,11]
def soma_points(future,world,calibration,offsets,root_offset,floor):
 rotations=r6(future[:,:,3:135].reshape(1,8,22,6));local=[]
 for j,parent in enumerate(model_parents):local.append(rotations[:,:,j] if parent<0 else local[parent]@rotations[:,:,j])
 R=world[:,:9].reshape(1,3,3);T=world[:,9:].reshape(1,1,1,3);axis=calibration[:,23];heading=calibration[:,22]
 root=torch.einsum('bij,btj->bti',axis@R,future[:,:,144:147])+torch.einsum('bij,btkj->btki',axis,T)[:,:,0]
 root=root+torch.cat((torch.zeros_like(root_offset),root_offset,torch.zeros_like(root_offset)),dim=-1)[:,None,:]
 global_rotations=[];points=[]
 for s,parent in enumerate(soma_parents):
  j=soma_slots[s]
  global_rotations.append((heading@axis@R)[:,None]@local[j]@calibration[:,None,j] if j>=0 else global_rotations[parent])
  points.append(root if parent<0 else points[parent]+(global_rotations[parent]@offsets[:,s,:,None][:,None]).squeeze(-1))
 result=torch.stack(points,dim=2);minimum=result[...,1].amin(dim=2,keepdim=True)-floor[:,None,1:2]
 correction=torch.stack((torch.zeros_like(minimum),minimum*floor[:,None,:1],torch.zeros_like(minimum)),dim=-1)
 return (result-correction).flatten(1)
def soma_points_grad(future,world,calibration,offsets,root_offset,floor,v):
 _,backward=torch.func.vjp(lambda f,w:soma_points(f,w,calibration,offsets,root_offset,floor),future,world);a,b=backward(v);return torch.cat((a.flatten(1),b),dim=1)
torch.manual_seed(1701)
world=torch.cat((torch.eye(3).reshape(1,9),torch.zeros(1,3)),dim=1)
delta=torch.tensor([[.0009881772566586733,-.375266432762146,.012849407270550728]])
cases=[('decoder-grad',decode_grad,(torch.randn(1,1,256),torch.randn(1,2,276),torch.randn(1,8,276)),['latent','history','cotangent'],'vae'),
 ('denoiser-grad',denoise_grad,(torch.randn(2,1,256),torch.tensor([4,4]),torch.randn(2,2,276),torch.randn(2,512),torch.randn(2,1,256)),['noise','timestep','history','text','cotangent'],'denoiser'),
 ('transition',transition,(torch.randn(1,8,276),world,delta),['future','world','delta'],None),
 ('transition-grad',transition_grad,(torch.randn(1,8,276),world,delta,torch.randn(1,1092)),['future','world','delta','cotangent'],None)]
calibration=torch.eye(3).reshape(1,1,3,3).repeat(1,24,1,1);offsets=torch.randn(1,30,3)*.1;root_offset=torch.zeros(1,1);floor=torch.zeros(1,2)
cases += [('soma-points',soma_points,(torch.randn(1,8,276),world,calibration,offsets,root_offset,floor),['future','world','calibration','offsets','root_offset','floor'],None),
 ('soma-points-grad',soma_points_grad,(torch.randn(1,8,276),world,calibration,offsets,root_offset,floor,torch.randn(1,720)),['future','world','calibration','offsets','root_offset','floor','cotangent'],None)]
fragments=[];pins={};fixtures=[]
for name,function,args,names,owner in cases:
 with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):module=make_fx(function,decomposition_table=core_aten_decompositions())(*args)
 path=base/(name+'-batch.onnx');torch.onnx.export(module,args,str(path),opset_version=17,input_names=names,output_names=['output'],do_constant_folding=False,dynamo=False)
 session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
 expected=function(*args).detach().numpy();actual=session.run(None,{k:v.numpy() for k,v in zip(names,args)})[0];error=float(np.max(np.abs(expected-actual)))
 assert np.allclose(expected,actual,atol=2e-4,rtol=2e-4),(name,error)
 fixtures.append({'name':name,'inputs':{k:{'shape':list(v.shape),'data':v.numpy().reshape(-1).tolist(),'integer':v.dtype==torch.int64} for k,v in zip(names,args)},'output':expected.reshape(-1).tolist()})
 model=onnx.load(path)
 for tensor in model.graph.initializer:
  data=tensor.raw_data;position=raw[owner].find(data) if owner else -1
  found=[e for e in entries if e[0]==owner and e[1]==position and e[2]==len(data) and not e[4]]
  if found:
   tensor.ClearField('raw_data');tensor.data_location=onnx.TensorProto.EXTERNAL
   for key,value in [('location','dart-weights.bin'),('offset',str(found[0][3])),('length',str(len(data)))]:entry=tensor.external_data.add();entry.key=key;entry.value=value
  else:
   values=onnx.numpy_helper.to_array(tensor)
   assert values.size<129 and np.all(np.isfinite(values)),('unknown large initializer',name,tensor.name,values.shape)
 binary=model.SerializeToString();path.write_bytes(binary);fragments.append(emit(name.replace('-','_'),model))
 pins[name]={'file':path.name,'bytes':len(binary),'sha256':hashlib.sha256(binary).hexdigest(),'nodes':len(model.graph.node),'opset':17,'maximumReferenceError':error}
 print(name,pins[name],flush=True)
output.write_text('// Generated native gradient/operator declarations; checkpoint tensors remain external.\n'+''.join(fragments))
(base/'gradient-graphs.json').write_text(json.dumps(pins,indent=2));(base/'gradient-reference.json').write_text(json.dumps(fixtures))
