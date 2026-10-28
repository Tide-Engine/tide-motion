"""Development-only dynamic-batch graph compiler. Uses the pinned DART export environment.
No weights or ONNX files are committed; native C++ constructs these graph declarations.
Pass the same arguments as dart_export.py, with --emit naming the batch C++ include.
"""
import sys,runpy,pathlib,hashlib,json
import numpy as np
import torch,onnx,onnxruntime as ort
from dart_graph_codegen import emit
arguments=sys.argv.copy()
output=pathlib.Path(arguments[arguments.index('--emit')+1])
sys.argv[sys.argv.index('--emit')+1]=str(output.parent/'dart-fixed-reference.inc')
g=runpy.run_path(str(pathlib.Path(__file__).with_name('dart_export.py')))
torch.set_num_threads(4)
base=g['base'];raw=g['raw'];entries=g['entries']
class Decoder(torch.nn.Module):
 def __init__(self):super().__init__();self.m=g['vae']
 def forward(self,latent,history):return self.m.decode(latent.permute(1,0,2),history,8,scale_latent=True)
class Policy(torch.nn.Module):
 def __init__(self):super().__init__();self.m=g['policy']
 def forward(self,observation):
  z=self.m.actor(self.m.get_embedding(observation))
  return torch.cat((torch.tanh(z)*4,self.m.actor_logstd.clamp(-1,1).exp().expand_as(z)),dim=1)
fragments=[];report={}
for name,model,values,inputs,out,axes in [
 ('denoiser',g['Den'](),(torch.randn(2,1,256),torch.tensor([9,9]),torch.randn(2,2,276),torch.randn(2,512)),['noise','timestep','history','text'],'x0',{k:{0:'batch'} for k in ['noise','timestep','history','text','x0']}),
 ('decoder',Decoder(),(torch.randn(2,1,256),torch.randn(2,2,276)),['latent','history'],'future',{k:{0:'batch'} for k in ['latent','history','future']}),
 ('policy',Policy(),(torch.randn(2,1069),),['observation'],'distribution',{k:{0:'batch'} for k in ['observation','distribution']})]:
 model.eval();p=base/(name+'-batch.onnx')
 torch.onnx.export(model,values,str(p),opset_version=17,input_names=inputs,output_names=[out],dynamic_axes=axes,do_constant_folding=False,dynamo=False)
 session=ort.InferenceSession(str(p),providers=['CPUExecutionProvider']);worst=0;fixtures=[]
 for n in [1,2,4,8,16]:
  data=[torch.randn(n,*v.shape[1:]) if v.dtype==torch.float32 else torch.full((n,),9,dtype=torch.int64) for v in values]
  with torch.no_grad():reference=model(*data).numpy()
  actual=session.run(None,{k:v.numpy() for k,v in zip(inputs,data)})[0]
  error=float(np.max(np.abs(reference-actual)));worst=max(worst,error)
  assert error<2e-4,(name,n,error)
  if n in [1,4]:fixtures.append({'batch':n,'inputs':[v.numpy().reshape(-1).tolist() for v in data],'output':reference.reshape(-1).tolist()})
 graph=onnx.load(p);owner={'decoder':'vae'}.get(name,name)
 for t in graph.graph.initializer:
  data=t.raw_data;position=raw[owner].find(data);half=False
  if position<0 and t.data_type==1:
   f=np.frombuffer(data,dtype='<f4');h=f.astype('<f2').tobytes()
   if np.array_equal(np.frombuffer(h,dtype='<f2').astype('<f4'),f):position=raw[owner].find(h);half=True
  found=[e for e in entries if e[0]==owner and e[1]==position and e[2]==len(data) and e[4]==half]
  assert found,(name,t.name,position,len(data))
  t.ClearField('raw_data');t.data_location=onnx.TensorProto.EXTERNAL
  for key,value in [('location','dart-weights.bin'),('offset',str(found[0][3])),('length',str(len(data)))]:
   item=t.external_data.add();item.key=key;item.value=value
  # No duplicated learned tensor file: batch graphs reference the existing pins.
 fragments.append(emit(name+'_batch',graph));binary=graph.SerializeToString();p.write_bytes(binary)
 report[name]={'file':p.name,'bytes':len(binary),'sha256':hashlib.sha256(binary).hexdigest(),'maximumReferenceError':worst,'opset':17,'nodes':len(graph.graph.node)}
 (base/(name+'-batch-reference.json')).write_text(json.dumps(fixtures))
 print(name,report[name],flush=True)
output.write_text('// Generated dynamic-batch operators; learned tensors use existing external offsets.\n'+''.join(fragments))
(base/'batch-graphs.json').write_text(json.dumps(report,indent=2))
