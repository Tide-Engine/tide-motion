"""Developer-only random or owner-authorized real checkpoint export/parity.

Not installed or used by Tide Motion. Source revision/hashes: momask-source-lock.json.
With no CHECKPOINT_DIRECTORY, CPU fixtures are random-weight architecture tests.
Usage: python tests/momask-export.py SOURCE_DIRECTORY IGNORED_OUTPUT_DIRECTORY [CHECKPOINT_DIRECTORY]
Real checkpoint mode verifies notes/momask-private-inputs.json before safe loading.
"""
import hashlib
import gzip
import importlib.util
import json
import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F

source, output = map(pathlib.Path, sys.argv[1:3])
checkpoint_root = pathlib.Path(sys.argv[3]) if len(sys.argv) == 4 else None
if checkpoint_root:
    pins = json.loads((pathlib.Path(__file__).parents[1] / 'notes/momask-private-inputs.json').read_text())
    for item in pins['files']:
        data = (checkpoint_root / item['file']).read_bytes()
        assert len(data) == item['size'] and hashlib.sha256(data).hexdigest() == item['sha256'], item['file']
output.mkdir(parents=True, exist_ok=False)
lock = json.loads((pathlib.Path(__file__).parents[1] / 'notes/momask-source-lock.json').read_text())
for record in lock['files']:
    data = (source / record['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
sys.path.insert(0, str(source))
# The exported steps consume precomputed text embeddings. Explicitly prevent
# constructor-triggered CLIP checkpoint downloads; this does NOT test CLIP.
sys.modules['clip'] = types.ModuleType('clip')
from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

class Mask(MaskTransformer):
    def load_and_freeze_clip(self, _):
        return nn.Identity()
    def forward(self, ids, text, padding):
        return self.trans_forward(ids, text, padding)

class Residual(ResidualTransformer):
    def load_and_freeze_clip(self, _):
        return nn.Identity()
    def forward(self, history, qids, text, padding):
        return self.output_project(self.trans_forward(history, qids, text, padding), qids - 1)

class Encoder(nn.Module):
    def __init__(self, vq):
        super().__init__()
        self.vq = vq
    def forward(self, rows):
        return self.vq.encode(rows)[0]

class Decoder(nn.Module):
    def __init__(self, vq):
        super().__init__()
        self.vq = vq
    def forward(self, ids):
        return self.vq.forward_decoder(ids)

torch.manual_seed(712)
torch.set_num_threads(4)
# Released opt.txt dimensions, independently inspected via bounded archive ranges;
# see notes/momask-configs.json and momask-archive-inventory.json.
opt = SimpleNamespace(num_tokens=512, num_quantizers=6, shared_codebook=False,
                      quantize_dropout_prob=0.2, mu=0.99)
vq = RVQVAE(opt, 263, 512, 512, 512, 2, 2, 512, 3, 3, 'relu', None).eval()
for layer in vq.quantizer.layers:
    layer.codebook.copy_(torch.randn_like(layer.codebook))
mask = Mask(512, 'text', 384, 1024, 8, 6, dropout=0.2, opt=opt).eval()
res = Residual(512, 'text', 384, 1024, 8, num_heads=6,
               dropout=0.2, cond_drop_prob=0.2, share_weight=True, opt=opt).eval()
res.requires_grad_(False)
if checkpoint_root:
    root = checkpoint_root / 't2m'
    vq.load_state_dict(torch.load(root / 'rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar', map_location='cpu', weights_only=True)['net'])
    mask.load_state_dict(torch.load(root / 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar', map_location='cpu', weights_only=True)['t2m_transformer'])
    res.load_state_dict(torch.load(root / 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw/model/net_best_fid.tar', map_location='cpu', weights_only=True)['res_transformer'])
res.process_embed_proj_weight()

def inputs(n):
    pad = torch.zeros(1, n, dtype=torch.bool)
    pad[:, -1] = True
    return {
        'masked_step': (torch.randint(0, 514, (1, n)), torch.randn(1, 512), pad),
        'residual_step': (torch.randn(1, n, 512), torch.tensor([3]), torch.randn(1, 512), pad),
        'rvq_encoder': (torch.randn(1, n * 4, 263),),
        'rvq_decoder': (torch.randint(0, 512, (1, n, 6)),),
    }

modules = {'masked_step': mask, 'residual_step': res,
           'rvq_encoder': Encoder(vq).eval(), 'rvq_decoder': Decoder(vq).eval()}
class PortableAttention(nn.MultiheadAttention):
    """Unfused self-attention with dynamic L, avoiding legacy export's fixed reshape."""
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        assert not is_causal and not need_weights
        length, batch, width = query.shape
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        def heads(x):
            return x.reshape(length, batch, self.num_heads, width // self.num_heads).permute(1, 2, 0, 3)
        q, k, v = map(heads, (q, k, v))
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            logits = logits + attn_mask
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]
            logits = logits.masked_fill(pad, float('-inf')) if pad.dtype == torch.bool else logits + pad
        attended = torch.matmul(logits.softmax(dim=-1), v).permute(2, 0, 1, 3).reshape(length, batch, width)
        return F.linear(attended, self.out_proj.weight, self.out_proj.bias), None

# Compare unfused decomposition to the original PyTorch attention before export.
decomposition_error = {}
torch.backends.mha.set_fastpath_enabled(False)
for name in ('masked_step', 'residual_step'):
    module = modules[name]
    trial = inputs(7)[name]
    with torch.no_grad():
        original = module(*trial)
    for layer in module.seqTransEncoder.layers:
        layer.self_attn.__class__ = PortableAttention
    with torch.no_grad():
        portable = module(*trial)
    torch.testing.assert_close(portable, original, atol=3e-5, rtol=3e-4)
    decomposition_error[name] = float((portable-original).abs().max())
report = {'purpose': 'random-weight CPU operator proof; no CLIP or pretrained inference',
          'source': lock['revision'], 'torch': torch.__version__, 'onnx': onnx.__version__,
          'ort': ort.__version__, 'attention_decomposition_max_abs': decomposition_error, 'components': {}}
for name, module in modules.items():
    args = inputs(7)[name]
    names = [f'input{i}' for i in range(len(args))]
    dynamic = {names[0]: {1: 'time'}}
    if name == 'masked_step':
        dynamic[names[2]] = {1: 'time'}
    if name == 'residual_step':
        dynamic[names[3]] = {1: 'time'}
    dynamic['output'] = {2 if name.endswith('step') else 1: 'out_time'}
    path = output / f'{name}.onnx'
    with torch.no_grad():
        # Disable fused attention fastpaths: export portable standard operators.
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(module, args, str(path), opset_version=17, dynamo=False,
                          input_names=names, output_names=['output'], dynamic_axes=dynamic)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    measurements = []
    for n in (4, 7, 49):
        trial = inputs(n)[name]
        with torch.no_grad():
            expected = module(*trial).numpy()
        actual = session.run(None, dict(zip(names, [a.numpy() for a in trial])))[0]
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        if name == 'rvq_encoder':
            assert np.array_equal(expected, actual), (name, n, error)
        else:
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
        fixtures = []
        for key, array in list(zip(names, [a.numpy() for a in trial])) + [('expected', expected)]:
            filename = f'{name}-{n}-{key}.bin'
            array.tofile(output / filename)
            fixtures.append({'name': key, 'file': filename, 'dtype': str(array.dtype), 'shape': list(array.shape)})
        measurements.append({'tokens': n, 'shape': list(actual.shape), 'max_abs': error, 'fixtures': fixtures})
    report['components'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'operators': sorted({node.op_type for node in model.graph.node}),
                                  'cases': measurements}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(name, [{k: v for k, v in case.items() if k != 'fixtures'} for case in measurements], flush=True)

if checkpoint_root:
    spec = importlib.util.spec_from_file_location('clip_model_source', source / 'clip-source/clip/model.py')
    clip_source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clip_source)
    clip = clip_source.build_model(torch.jit.load(str(checkpoint_root / 'clip.pt'), map_location='cpu').state_dict()).float().eval()
    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = clip
        def forward(self, tokens):
            return self.clip.encode_text(tokens)
    text = Text().eval()
    text_ids = torch.zeros(1, 77, dtype=torch.int64)
    text_ids[0, :4] = torch.tensor([49406, 320, 2533, 49407])
    with torch.no_grad():
        text_reference = text(text_ids)
    for layer in clip.transformer.resblocks:
        layer.attn.__class__ = PortableAttention
    with torch.no_grad():
        torch.testing.assert_close(text(text_ids), text_reference, atol=1e-4, rtol=1e-3)
        torch.onnx.export(text, (text_ids,), str(output / 'text.onnx'), opset_version=17, dynamo=False,
                          input_names=['input0'], output_names=['output'])
    session = ort.InferenceSession(str(output / 'text.onnx'), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'input0': text_ids.numpy()})[0]
    np.testing.assert_allclose(actual, text_reference.numpy(), atol=1e-4, rtol=1e-3)
    text_ids.numpy().tofile(output / 'text-77-input0.bin')
    text_reference.numpy().tofile(output / 'text-77-expected.bin')
    report['components']['text'] = {'sha256': hashlib.sha256((output/'text.onnx').read_bytes()).hexdigest(),
        'cases': [{'tokens':77, 'fixtures':[
            {'name':'input0','file':'text-77-input0.bin','dtype':'int64','shape':[1,77]},
            {'name':'expected','file':'text-77-expected.bin','dtype':'float32','shape':[1,512]}]}]}
    report['purpose'] = 'Real checkpoint component parity; owner-authorized private experiment, rights unresolved'
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    embeddings = res.token_embed_weight.detach().numpy().astype('<f4')
    embeddings.tofile(output / 'res_embeddings.bin')
    # Official byte-BPE vocabulary and merges. Runtime initially supports ASCII
    # prompts only; this excludes ftfy/Unicode equivalence claims.
    # Compute precisely the official bytes_to_unicode ordering.
    byte_values = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    characters = byte_values.copy(); extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value); characters.append(256+extra); extra += 1
    vocab = [chr(c) for c in characters]
    merges = [line.split() for line in gzip.open(checkpoint_root/'bpe.txt.gz', 'rt', encoding='utf-8').read().split('\n')[1:49152-256-2+1]]
    vocab += [v+'</w>' for v in vocab]
    vocab += [''.join(pair) for pair in merges]
    vocab += ['<|startoftext|>', '<|endoftext|>']
    meta = {'schema':1, 'status':pins['status'], 'vocabulary':vocab, 'merges':merges,
            'mean':np.load(root/'rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/mean.npy').tolist(),
            'std':np.load(root/'rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/std.npy').tolist()}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False),encoding='utf-8')
    bundle = {'schema':1, 'rights':pins['status'], 'files':{}}
    for file in list(output.glob('*.onnx')) + [output/'metadata.json', output/'res_embeddings.bin']:
        bundle['files'][file.name] = {'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'size':file.stat().st_size}
    (output/'bundle.json').write_text(json.dumps(bundle,indent=2))
    print('Real weights and CLIP exported; all assets remain private.', flush=True)
