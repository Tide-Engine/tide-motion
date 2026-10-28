"""Developer-only ARDY conversion/reference tool. Never called by tide-motion.

Uses the pinned NVIDIA source and commercially licensed SOMA KimodoRep weights.
All generated weights/fixtures belong outside Git. See notes/ardy-feasibility.md.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
import torch

SOURCE = "e706e27acc88070918308a10922b8f7174d0825e"
REVISION = "e6c6804fb6b55636ad927145d74959fd46882ccd"
TOKENS = 75
FRAMES = TOKENS * 4


class Denoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.denoiser
        self.rep = model.motion_rep

    def forward(self, x, text, observed, mask, lengths, timestep, heading, guidance):
        h, g, f = lengths[0:1], lengths[1:2], lengths[2:3]
        indices = torch.arange(FRAMES)[None]
        hm, gm = indices < h[:, None], (indices >= h[:, None]) & (indices < (h+g)[:, None])
        fm = (indices >= (h+g)[:, None]) & (indices < (h+g+f)[:, None])
        ht = hm.reshape(1,TOKENS,4).all(-1)
        gt = gm.reshape(1,TOKENS,4).all(-1)
        ft = fm.reshape(1,TOKENS,4).all(-1) & mask.reshape(1,TOKENS,4,369).any(-1).any(-1)
        obs = self.rep.normalize(observed) * mask
        return self.net(guidance[0:1], guidance[1:2], x, h, g, f,
                        hm.float(), gm.float(), fm.float(), ht.float(), gt.float(), ft.float(),
                        text, torch.ones(1,1), timestep, heading, mask, obs)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ae = model.autoencoder
        self.rep = model.motion_rep

    def forward(self, motion, length):
        valid = torch.arange(FRAMES)[None] < length[:,None]
        normalized = self.rep.normalize(motion)
        body = self.ae.tokenize(normalized, valid)
        return torch.cat([normalized[:,:,:5].reshape(1,TOKENS,20),body],-1)


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ae = model.autoencoder
        self.rep = model.motion_rep

    def forward(self, x, length):
        root = x[:,:,:20].reshape(1,FRAMES,5)
        local = self.rep.global_root_to_local_root(root, normalized=True, lengths=length)
        mask = torch.arange(FRAMES)[None] < length[:,None]
        output = self.ae.detokenize(x[:,:,20:],local,mask)
        return self.rep.unnormalize(torch.cat([root,output['body']],-1))


def digest(path):
    return hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=pathlib.Path,required=True)
    p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    args=p.parse_args()
    from prepare import verify
    verify(args.checkpoint)
    assert subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()==SOURCE
    assert digest(args.checkpoint/'denoiser.safetensors')=='532ead2c40f27733556c099e8d77e8f227e4047b5a88d556b995d0b9c012dec7'
    assert digest(args.checkpoint/'tokenizer.safetensors')=='c156290ca9da2f821d7e1570aad89f153b943e4bdabb4451eef92ba3fb385398'
    sys.path.insert(0,str(args.source.resolve()))
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    conf=OmegaConf.load(args.checkpoint/'config.yaml')
    conf=OmegaConf.merge(conf, {'checkpoint_dir':str(args.checkpoint.resolve())})
    cfg=OmegaConf.to_container(conf,resolve=True)
    cfg.pop('checkpoint_dir')
    model=instantiate(cfg,text_encoder=None,device='cpu')
    model.eval()
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(431)
    args.output.mkdir(parents=True,exist_ok=True)
    x=torch.randn(1,TOKENS,148)
    text=torch.randn(1,1,4096)
    obs=torch.zeros(1,FRAMES,369)
    mask=torch.zeros_like(obs)
    lengths=torch.tensor([4,60,FRAMES-64])
    step=torch.tensor([9])
    heading=torch.tensor([0.25])
    guidance=torch.tensor([2.,2.])
    modules={'denoiser':(Denoiser(model),(x,text,obs,mask,lengths,step,heading,guidance),
                         ['x','text','observed','mask','lengths','timestep','heading','guidance']),
             'encoder':(Encoder(model),(obs,torch.tensor([64])),['motion','length']),
             'decoder':(Decoder(model),(x,torch.tensor([64])),['x','length'])}
    import onnxruntime as ort
    results={}
    for name,(net,inputs,names) in modules.items():
        net.eval()
        print('Exporting',name,flush=True)
        path=args.output/(name+'.onnx')
        with torch.no_grad():
            torch.onnx.export(net,inputs,str(path),input_names=names,output_names=['output'],
                              opset_version=17,do_constant_folding=True,dynamo=False)
        opts=ort.SessionOptions();opts.intra_op_num_threads=8
        session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        errs=[]
        for history in (0,4,240):
            vals=[v.clone() for v in inputs]
            if name=='denoiser':
                vals[4]=torch.tensor([history,60,FRAMES-history-60])
                vals[2][:,history+40,0]=0.8
                vals[3][:,history+40,0]=1
                vals[0][:,:,:20]*=0.1
            else:
                vals[1]=torch.tensor([max(4,history)])
            with torch.no_grad(): ref=net(*vals).numpy()
            feed={n:v.numpy() for n,v in zip(names,vals) if n in {i.name for i in session.get_inputs()}}
            actual=session.run(None,feed)[0]
            err=np.abs(ref-actual)
            assert np.isfinite(actual).all()
            print(name,history,'max',err.max(),'mean',err.mean(),flush=True)
            # Raw decoder features include velocities with large magnitude.
            assert np.allclose(ref,actual,rtol=2e-3,atol=2e-3), (name,err.max())
            errs.append({'history':history,'max':float(err.max()),'mean':float(err.mean())})
            # Flat JSON fixtures permit independent native ORT and host checks.
            fixture={'inputs':{n:{'shape':list(v.shape),'dtype':str(v.numpy().dtype),'data':v.flatten().tolist()}
                                for n,v in zip(names,vals) if n in feed},
                     'output':{'shape':list(ref.shape),'data':ref.flatten().tolist()}}
            (args.output/f'{name}-{history}.json').write_text(json.dumps(fixture))
        results[name]=errs
    def values(stat,field):return getattr(stat,field).detach().flatten().tolist()
    meta={'schema':'tide-ardy-1','sourceRevision':SOURCE,'checkpointRevision':REVISION,
          'tokens':TOKENS,'frames':FRAMES,'horizon':60,'fps':30,
          'rootMean':values(model.motion_rep.global_root_stats,'mean'),
          'rootScale':values(model.motion_rep.global_root_stats,'std_eps'),
          'latentMean':values(model.autoencoder.post_quantization_stats,'mean'),
          'latentScale':values(model.autoencoder.post_quantization_stats,'std_eps'),
          'parity':results,'files':{}}
    for n in modules:
        path=args.output/(n+'.onnx');meta['files'][path.name]={'bytes':path.stat().st_size,'sha256':digest(path)}
    meta['skeleton']={'names':model.skeleton.joint_names if hasattr(model.skeleton,'joint_names') else []}
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    import shutil
    shutil.copyfile(args.checkpoint/'LICENSE',args.output/'LICENSE')
    (args.output/'NOTICE').write_text('Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n'
                                    'Converted by Tide Motion from ARDY SOMA KimodoRep at '+REVISION+'\n')
    print('PASS complete network export/reference checks',flush=True)


if __name__=='__main__':main()
