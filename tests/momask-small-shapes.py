"""Development-only PyTorch/ORT parity for newly used 1/2/3-token observations.
Usage: python tests/momask-small-shapes.py SOURCE CHECKPOINTS BUNDLE NEW_FIXTURE_DIR
No downloads, model publication or runtime Python dependency.
"""
import hashlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

source, checkpoints, bundle, output = map(Path,sys.argv[1:])
root = Path(__file__).parents[1]
for item in json.loads((root/'notes/momask-private-inputs.json').read_text())['files']:
    data = (checkpoints/item['file']).read_bytes()
    assert len(data)==item['size'] and hashlib.sha256(data).hexdigest()==item['sha256']
for item in json.loads((root/'notes/momask-source-lock.json').read_text())['files']:
    assert hashlib.sha256((source/item['path']).read_bytes()).hexdigest()==item['sha256']
output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(source)); sys.modules['clip']=types.ModuleType('clip')
from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

class Mask(MaskTransformer):
    def load_and_freeze_clip(self,_): return nn.Identity()
    def forward(self,ids,text,padding): return self.trans_forward(ids,text,padding)

class Residual(ResidualTransformer):
    def load_and_freeze_clip(self,_): return nn.Identity()
    def forward(self,history,layer,text,padding):
        return self.output_project(self.trans_forward(history,layer,text,padding),layer-1)

torch.set_num_threads(4); torch.manual_seed(1729); torch.backends.mha.set_fastpath_enabled(False)
opt=SimpleNamespace(num_tokens=512,num_quantizers=6,shared_codebook=False,quantize_dropout_prob=.2,mu=.99)
vq=RVQVAE(opt,263,512,512,512,2,2,512,3,3,'relu',None).eval()
mask=Mask(512,'text',384,1024,8,6,dropout=.2,opt=opt).eval()
res=Residual(512,'text',384,1024,8,num_heads=6,dropout=.2,cond_drop_prob=.2,share_weight=True,opt=opt).eval()
def load(file,key): return torch.load(checkpoints/'t2m'/file,map_location='cpu',weights_only=True)[key]
vq.load_state_dict(load('rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar','net'))
mask.load_state_dict(load('t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar','t2m_transformer'))
res.load_state_dict(load('tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw/model/net_best_fid.tar','res_transformer'))
res.process_embed_proj_weight()
modules={'masked_step':mask,'residual_step':res,'rvq_encoder':lambda x:vq.encode(x)[0],'rvq_decoder':vq.forward_decoder}
report={'purpose':'real checkpoint small observation tensor parity','components':{}}
options=ort.SessionOptions(); options.intra_op_num_threads=4
for name,module in modules.items():
    session=ort.InferenceSession(str(bundle/(name+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
    cases=[]
    for n in (1,2,3):
        padding=torch.zeros(1,n,dtype=torch.bool)
        args={'masked_step':(torch.randint(0,512,(1,n)),torch.randn(1,512),padding),
              'residual_step':(torch.randn(1,n,512),torch.tensor([3]),torch.randn(1,512),padding),
              'rvq_encoder':(torch.randn(1,n*4,263),),
              'rvq_decoder':(torch.randint(0,512,(1,n,6)),)}[name]
        if name=='residual_step' and n==1:
            args=(torch.cat([args[0],torch.zeros(1,1,512)],dim=1),args[1],args[2],torch.tensor([[False,True]]))
        with torch.no_grad(): expected=module(*args).numpy()
        actual=session.run(None,{f'input{i}':a.numpy() for i,a in enumerate(args)})[0]
        if expected.dtype==np.int64: np.testing.assert_array_equal(actual,expected)
        else: np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
        fixtures=[]
        for key,array in [*(('input'+str(i),a.numpy()) for i,a in enumerate(args)),('expected',expected)]:
            file=f'{name}-{n}-{key}.bin'; array.tofile(output/file)
            fixtures.append(dict(name=key,file=file,dtype=str(array.dtype),shape=list(array.shape)))
        error=float(np.max(np.abs(actual-expected)))
        cases.append(dict(tokens=n,fixtures=fixtures,max_abs=error))
        print(name,n,error,flush=True)
    report['components'][name]={'cases':cases}
(output/'report.json').write_text(json.dumps(report,indent=2))
