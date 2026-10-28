"""Developer-only random-weight operator proof. Never loads pretrained artifacts.

Not installed or used by Tide Motion. Source revision/hashes: momask-source-lock.json.
CPU fixtures are architectural tests, not pretrained numerical/quality evidence.
Usage: python tests/momask-export.py SOURCE_DIRECTORY IGNORED_OUTPUT_DIRECTORY
"""
import hashlib
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

source, output = map(pathlib.Path, sys.argv[1:])
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
        assert attn_mask is None and not is_causal and not need_weights
        length, batch, width = query.shape
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        def heads(x):
            return x.reshape(length, batch, self.num_heads, width // self.num_heads).permute(1, 2, 0, 3)
        q, k, v = map(heads, (q, k, v))
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
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
