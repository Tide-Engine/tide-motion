"""Developer-only independent native sampling/tokenizer/GLB checks, no runtime dependency.
Usage: python tests/momask-reference.py SOURCE BUNDLE NATIVE.glb [EDIT.glb BEGIN END]
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

source, bundle, glb = map(Path, sys.argv[1:4])
recipe = json.loads(glb.with_suffix('.glb.json').read_text())
spec = importlib.util.spec_from_file_location('official_tokenizer', source/'clip-source/clip/simple_tokenizer.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
# The BPE is a pinned developer input, kept beside the private checkpoint root.
tokenizer = module.SimpleTokenizer(str(bundle.parent/'momask-live/bpe.txt.gz'))
ids = [49406] + tokenizer.encode(recipe['prompt']) + [49407]
ids = ids[:77]
if len(tokenizer.encode(recipe['prompt']))+2 > 77:
    ids[-1] = 49407
ids += [0]*(77-len(ids))
assert ids == recipe['textTokens'], 'Native ASCII byte-BPE differs from official tokenizer'

sessions = {name: ort.InferenceSession(str(bundle/(name+'.onnx')), providers=['CPUExecutionProvider'])
            for name in ('text','masked_step','residual_step','rvq_decoder')}
text = sessions['text'].run(None, {'input0':np.array([ids],dtype=np.int64)})[0]
embeddings = np.fromfile(bundle/'res_embeddings.bin',dtype='<f4').reshape(5,513,512)
size = len(recipe['baseTokens']); zero = np.zeros((1,512),dtype=np.float32)
padding = np.zeros((1,size),dtype=bool)
state = recipe['seed']; mask64 = 2**64-1
def uniform():
    global state
    state = (state+0x9e3779b97f4a7c15)&mask64
    z = state; z = ((z^(z>>30))*0xbf58476d1ce4e5b9)&mask64
    z = ((z^(z>>27))*0x94d049bb133111eb)&mask64; z ^= z>>31
    return min(((z>>11)+.5)/2**53,1-2**-53)

def guidance(name, inputs, scale):
    inputs = dict(inputs)
    text_name = 'input2' if name == 'residual_step' else 'input1'
    inputs[text_name] = text
    c = sessions[name].run(None, inputs)[0][0]
    inputs[text_name] = zero
    u = sessions[name].run(None, inputs)[0][0]
    return (u+(c-u).astype(np.float64)*scale).astype(np.float32)

def sample(logits, t, gumbel=False):
    vocab = len(logits); order = np.argsort(-logits[:,t],kind='stable')
    k = max(1,math.ceil((1-recipe['topkr'])*vocab)); keep = set(order[:k])
    values = logits[:,t].astype(np.float64)/recipe['temperature']
    if gumbel:
        noises = np.array([-math.log(-math.log(uniform())) for _ in range(vocab)])
        return max(sorted(keep), key=lambda c:values[c]+noises[c])
    high = values[order[0]]; probabilities = [math.exp(values[c]-high) if c in keep else 0 for c in range(vocab)]
    choice = uniform()*sum(probabilities)
    for c, probability in enumerate(probabilities):
        choice -= probability
        if choice < 0:
            return c
    return int(order[k-1])

tokens = np.full((1,size),512,dtype=np.int64); scores = np.zeros(size)
for iteration in range(recipe['steps']):
    t = np.float32(iteration/(recipe['steps']-1))
    angle = np.float32(np.float32(t*np.float32(math.pi))*np.float32(.5))
    count = max(1,round(float(np.float32(np.float32(math.cos(float(angle)))*np.float32(size)))))
    selected = np.argsort(scores,kind='stable')[:count]; tokens[0,selected] = 512
    logits = guidance('masked_step',{'input0':tokens,'input2':padding},recipe['guidance'])
    predictions = np.array([sample(logits,t) for t in range(size)])
    tokens[0,selected] = predictions[selected]
    scores[:] = 1e5
    for t in selected:
        column = logits[:,t].astype(np.float64); high = max(column)
        scores[t] = math.exp(column[predictions[t]]-high)/sum(math.exp(v-high) for v in column)
assert tokens[0].tolist() == recipe['baseTokens'], 'Masked sampling differs from native'
history = np.zeros((1,size,512),dtype=np.float32); all_ids = np.zeros((size,6),dtype=np.int64); all_ids[:,0] = tokens[0]
for layer in range(1,6):
    history += embeddings[layer-1,tokens]
    logits = guidance('residual_step',{'input0':history,'input1':np.array([layer],dtype=np.int64),'input3':padding},recipe['residualGuidance'])
    tokens[0] = [sample(logits,t,True) for t in range(size)]
    all_ids[:,layer] = np.where(tokens[0]==512,-1,tokens[0])
assert all_ids.reshape(-1).tolist() == recipe['tokens'], 'Residual sampling differs from native'
print('Official tokenizer and all native masked/residual token choices match independent reference.')

if len(sys.argv) == 7:
    spec = importlib.util.spec_from_file_location('soma_metrics', Path(__file__).with_name('soma_metrics.py'))
    metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    def tracks(path):
        import struct
        doc, binary = metrics.payload(path); output = {}
        for channel in doc['animations'][0]['channels']:
            target = channel['target']; sampler = doc['animations'][0]['samplers'][channel['sampler']]
            accessor = doc['accessors'][sampler['output']]; view = doc['bufferViews'][accessor['bufferView']]
            width = {'VEC3':3,'VEC4':4}[accessor['type']]; offset = view.get('byteOffset',0)+accessor.get('byteOffset',0)
            data = np.frombuffer(binary,dtype='<f4',count=accessor['count']*width,offset=offset).reshape(-1,width)
            output[(doc['nodes'][target['node']]['name'],target['path'])] = data
        return output
    edited = Path(sys.argv[4]); begin,end = map(int,sys.argv[5:]); a,b = tracks(glb),tracks(edited)
    worst = 0.0; exact = True
    for key,value in a.items():
        assert value.shape == b[key].shape
        other = b[key].copy()
        if key[1] == 'rotation':
            # Tide exports shortest-sign continuity. q and -q encode the same
            # pose, and a changed interior can change subsequent key signs.
            other[np.sum(value*other,axis=1)<0] *= -1
        fixed = np.ones(len(value),dtype=bool); fixed[begin:end] = False; fixed[begin] = fixed[end-1] = True
        delta = np.max(np.abs(value[fixed]-other[fixed])); worst = max(worst,float(delta)); exact &= np.array_equal(value[fixed],other[fixed])
        assert delta < 2e-6, (key,delta)
    print('Exported outside-range and endpoint components: exact=',exact,'max error=',worst)
