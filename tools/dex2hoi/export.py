"""Developer-only conversion of manually supplied Dex2HOI HUMOTO EMA weights.

No downloads; never invoked by the installed executable. Requires the pinned
upstream checkout and a previously validated canonical CLIP ONNX/tokenizer bundle.
All generated graphs/fixtures are private model assets, not release artifacts.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

REVISION = 'a92aca6b48b7a0c9a45a1465aa984213d3c8ceda'
CLIP_INPUTS = {
    'text.onnx': 'b8bf1b755f38d930a4eb320b8fd1bc480d3e31a5d7836fc244c49155bd374ada',
    'metadata.json': 'dafef7d56c6c3e639066abe04dedbd07d3ee94de342f31e48a68ef6315e79dc3',
}


def digest(path):
    return {'size': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--weights', type=pathlib.Path, required=True)
    parser.add_argument('--clip-bundle', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Unsupported upstream revision')
    lock = json.loads((pathlib.Path(__file__).with_name('source-lock.json')).read_text())
    for name, record in lock['files'].items():
        if digest(args.source / name) != record:
            raise ValueError('Modified upstream source: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / 'bundle'
    fixtures = args.output / 'fixtures'
    bundle.mkdir()
    fixtures.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(20261028)
    torch.backends.mha.set_fastpath_enabled(False)

    # Constructor-only adapters for dependencies unused by the trained BPS branch.
    # Executing PointNet through these adapters is expressly unsupported.
    clip_stub = types.ModuleType('clip')
    clip_stub.load = lambda *a, **k: (nn.Identity(), None)
    clip_stub.model = types.SimpleNamespace(convert_weights=lambda _: None)
    sys.modules['clip'] = clip_stub
    pyg = types.ModuleType('torch_geometric')
    pyg.data = types.SimpleNamespace(Batch=object)
    sys.modules['torch_geometric'] = pyg
    def forbidden(*a, **k):
        raise RuntimeError('PointNet is not supported by this BPS exporter')
    pointwise = types.ModuleType('model.object_conditioning.object_encoder_pointwise')
    pointwise.ObjectPointwiseEncoder = forbidden
    pointwise.pyg_batch_to_torch_batch = lambda data, name: getattr(data, name).reshape(2, 128, -1)
    sys.modules[pointwise.__name__] = pointwise
    global_encoder = types.ModuleType('model.object_conditioning.object_encoder_global')
    global_encoder.ObjectEncoder = forbidden
    sys.modules[global_encoder.__name__] = global_encoder
    sys.path.insert(0, str(args.source.resolve()))
    # Upstream package initialization configures plotting, which is unrelated to
    # these graphs. Resolve its submodules without running that optional UI hook.
    package = types.ModuleType('model')
    package.__path__ = [str((args.source/'model').resolve())]
    sys.modules['model'] = package
    from model.hoi.hoi_motion_fusion import HOIMotionFusion, TwoObjInterGenConfig

    config = TwoObjInterGenConfig(num_objects=2, obj_slot_size=18,
        cross_attn_layers=8, x0_as_condition=True, input_process_hidden_dim=1024,
        fusion_type='transformer', fusion_transformer_num_heads=3)
    model = HOIMotionFusion(pred_len=60, context_len=10, n_feats_multi=429,
        num_layers=8, cond_mask_prob=0.1, config=config, obj_encoder_type='bps').eval()
    checkpoint = torch.load(args.weights / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_avg'], strict=True)
    model.requires_grad_(False)
    mean = torch.load(args.weights / 'mean.pt', weights_only=True, map_location='cpu').reshape(-1)
    std = torch.load(args.weights / 'std.pt', weights_only=True, map_location='cpu').reshape(-1)
    eps = float(torch.load(args.weights / 'eps.pt', weights_only=True, map_location='cpu'))
    if mean.shape != (429,) or std.shape != (429,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid normalization')
    (bundle / 'normalization.json').write_text(json.dumps({'schema': 1, 'mean': mean.tolist(), 'std': std.tolist(), 'eps': eps}), encoding='utf-8')

    class ObjectGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.object_encoder
            self.spatial = model.spatial_encoding
            self.project = model.embed_obj
        def forward(self, points):
            data = types.SimpleNamespace(pos=points.reshape(-1, 3))
            embeddings, basis = self.encoder(data)
            return self.project(torch.cat((embeddings, self.spatial(basis)), dim=-1))

    class InitGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = model.init_tfm_encoder.linear
        def forward(self, features):
            return self.linear(features)

    class AlreadyProjected(nn.Module):
        def forward(self, x):
            return x[..., :512]

    obj_graph = ObjectGraph().eval()
    points = torch.randn(2, 128, 3) * 0.1
    x = torch.randn(1, 429, 1, 60)
    timestep = torch.tensor([499], dtype=torch.int64)
    prefix = torch.randn(1, 429, 1, 10)
    text = torch.randn(1, 512)
    object_mask = torch.tensor([True, True])
    root_x0 = torch.tensor([[0.2, 0.9, -0.1]])
    obj_x0 = torch.tensor([[[0.3, 1.0, 0.1], [-0.3, 1.0, 0.1]]])
    tfm = torch.eye(4).unsqueeze(0)
    tfm[:, :3, 3] = root_x0
    init_features = torch.cat((model.init_tfm_encoder.spatial_encoding(root_x0),
        torch.tensor([[1., 0., 0., 0., 1., 0.]]), torch.zeros(1, 3)), dim=-1)
    with torch.no_grad():
        raw_embeddings, basis = model.object_encoder(types.SimpleNamespace(pos=points.reshape(-1, 3)))
        objects = obj_graph(points)
        reference_y = {'prefix': prefix, 'text_embed': (text.unsqueeze(0), torch.zeros(1, 1, dtype=torch.bool)),
            'obj_emb_list': [(raw_embeddings[i:i+1], basis[i:i+1]) for i in range(2)],
            'obj_mask': object_mask.reshape(1, 2), 'tfms_root_global': tfm,
            'obj_x0': obj_x0}
        full_reference = model(x, timestep, reference_y)

    # Move invariant object affine/spatial work out of every denoising iteration.
    # The cached tensor is supplied in place of its first 512 input components;
    # original stream type embeddings and masks are still applied by the network.
    model.embed_obj = AlreadyProjected()
    model.init_tfm_encoder = InitGraph()

    class DenoiserGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = model
        def forward(self, x, timestep, prefix, text, objects, object_mask,
                    init_features, obj_x0, text_scale):
            y = {'prefix': prefix,
                'text_embed': ((text * text_scale.reshape(1, 1)).unsqueeze(0), torch.zeros(1, 1, dtype=torch.bool)),
                'obj_emb_list': [objects[i:i+1] for i in range(2)],
                'obj_mask': object_mask.reshape(1, 2), 'tfms_root_global': init_features,
                'obj_x0': obj_x0}
            return self.network(x, timestep, y)

    graph = DenoiserGraph().eval()
    # Pinned HUMOTO's dataset/collate supplies obj_x0 but no root_x0 network
    # token. Its runner recovers root_x0 from tfms_root_global for host decode
    # only. Adding a root token here would silently change learned conditioning.
    inputs = (x, timestep, prefix, text, objects, object_mask, init_features, obj_x0, torch.ones(1))
    names = ['x', 'timestep', 'prefix', 'text', 'objects', 'object_mask', 'init_features', 'obj_x0', 'text_scale']
    with torch.no_grad():
        split_reference = graph(*inputs)
    torch.testing.assert_close(split_reference, full_reference, atol=2e-5, rtol=2e-4)
    report = {'sourceRevision': REVISION, 'state': 'model_avg',
              'decompositionMaxAbsError': float((split_reference-full_reference).abs().max()), 'cases': []}

    def export(module, example, path, input_names, output_name):
        with torch.no_grad():
            torch.onnx.export(module, example, str(path), opset_version=17, dynamo=False,
                input_names=input_names, output_names=[output_name])
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    obj_session = export(obj_graph, (points,), bundle/'object.onnx', ['points'], 'objects')
    obj_actual = obj_session.run(None, {'points': points.numpy()})[0]
    np.testing.assert_allclose(obj_actual, objects.numpy(), atol=1e-4, rtol=1e-3)
    session = export(graph, inputs, bundle/'denoiser.onnx', names, 'prediction')
    for index, (time, active, scale) in enumerate([(499, [True, True], 1.), (250, [True, False], 0.), (0, [True, True], 1.)]):
        trial = list(inputs)
        trial[1] = torch.tensor([time], dtype=torch.int64)
        trial[5] = torch.tensor(active)
        trial[-1] = torch.tensor([scale])
        with torch.no_grad():
            expected = graph(*trial).numpy()
        actual = session.run(None, {name: value.numpy() for name, value in zip(names, trial)})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-3)
        case = {'timestep': time, 'active': active, 'textScale': scale,
                'maxAbsError': float(np.max(np.abs(actual-expected)))}
        report['cases'].append(case)
        fixture = {name: {'shape': list(value.shape), 'dtype': str(value.numpy().dtype), 'data': value.numpy().reshape(-1).tolist()} for name, value in zip(names, trial)}
        fixture['expected'] = {'shape': list(expected.shape), 'dtype': 'float32', 'data': expected.reshape(-1).tolist()}
        (fixtures/f'denoise-{index}.json').write_text(json.dumps(fixture), encoding='utf-8')
        print(case, flush=True)

    # Use the owner's existing canonical CLIP graph, verifying its source bundle.
    clip_manifest = json.loads((args.clip_bundle/'bundle.json').read_text())
    for filename in ['text.onnx', 'metadata.json']:
        record = clip_manifest['files'][filename]
        actual = digest(args.clip_bundle/filename)
        if actual['sha256'] != record['sha256'] or actual['size'] != record['size']:
            raise ValueError('CLIP source bundle does not match its manifest: '+filename)
        if actual['sha256'] != CLIP_INPUTS[filename]:
            raise ValueError('Unsupported canonical ViT-B/32 source asset: '+filename)
    clip_graph = onnx.load(str(args.clip_bundle/'text.onnx'))
    renames = {clip_graph.graph.input[0].name: 'tokens', clip_graph.graph.output[0].name: 'text'}
    for node in clip_graph.graph.node:
        for i, value in enumerate(node.input):
            node.input[i] = renames.get(value, value)
        for i, value in enumerate(node.output):
            node.output[i] = renames.get(value, value)
    for value in list(clip_graph.graph.input)+list(clip_graph.graph.output):
        value.name = renames.get(value.name, value.name)
    onnx.save(clip_graph, str(bundle/'clip.onnx'))
    onnx.checker.check_model(str(bundle/'clip.onnx'))
    metadata = json.loads((args.clip_bundle/'metadata.json').read_text(encoding='utf-8'))
    (bundle/'tokenizer.json').write_text(json.dumps({'schema':1, 'vocabulary':metadata['vocabulary'], 'merges':metadata['merges']}, ensure_ascii=False), encoding='utf-8')
    for source, filename in [
        (args.source/'model/diffusion_planner/LICENSE', 'MDM-LICENSE.txt'),
        (args.source/'model/diffusion_planner/utils/PYTORCH3D_LICENSE', 'PYTORCH3D-LICENSE.txt'),
        (pathlib.Path(__file__).resolve().parents[2]/'licenses/OPENAI-CLIP.txt', 'OPENAI-CLIP-LICENSE.txt'),
    ]:
        shutil.copyfile(source, bundle/filename)
    (bundle/'ARTIFACT-TERMS.txt').write_text(
        'User-supplied private conversion. Top-level Dex2HOI and checkpoint grants remain unverified.\n'
        'Included component notices retain their original scope; they do not establish rights to all artifacts.\n'
        'Tide does not bundle or automatically download these model files.\n', encoding='utf-8')
    manifest = {'schema':1, 'backend':'dex2hoi', 'upstreamRevision':REVISION,
        'representation':{'bones':65, 'numObjects':2, 'objectSlotSize':18, 'features':429,
            'contextFrames':10, 'predictionFrames':60, 'fps':30},
        'architecture':{'objectEncoder':'bps','basisPoints':512,'latentDim':512,'decoderLayers':8,
            'crossAttentionLayers':8,'hiddenDim':1024,'fusion':'transformer','fusionHeads':3},
        'diffusion':{'steps':500,'schedule':'cosine','prediction':'x0','eta':0},
        'origin':{name:digest(args.weights/filename) for name,filename in [('checkpoint','model.pt'),('mean','mean.pt'),('std','std.pt'),('eps','eps.pt')]},
        'licensingStatus':'user-supplied-unverified', 'files':{}}
    manifest['origin']['state'] = 'model_avg'
    for path in sorted(bundle.iterdir()):
        manifest['files'][path.name] = digest(path)
    (bundle/'bundle.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report['objectMaxAbsError'] = float(np.max(np.abs(obj_actual-objects.numpy())))
    report['clipSource'] = {name:digest(args.clip_bundle/name) for name in ['text.onnx','metadata.json']}
    (args.output/'parity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Private native bundle:', bundle, flush=True)


if __name__ == '__main__':
    main()
