"""Emit readable native protobuf/operator declarations; never encode model files in Git."""
import json,struct,math
import numpy as np
from google.protobuf.descriptor import FieldDescriptor as F

def raw(data):
    if len(data)>128:
        if not any(data):return f'dz({len(data)})'
        assert len(data)%4==0
        a=np.frombuffer(data,dtype='<f4');n=math.isqrt(a.size)
        expected=np.where(np.triu(np.ones((n,n),dtype=bool),1),-np.inf,0).astype('<f4')
        assert n*n==a.size and np.array_equal(a.reshape(n,n),expected),'unexpected non-mathematical graph constant'
        return f'dc({n})'
    return 'dr({'+','.join(str(v) for v in data)+'})'

def integer(value):return str(int(value)&((1<<64)-1))+'ULL'

def packed(fd,values):
    data=bytearray()
    for v in values:
        if fd.type==F.TYPE_FLOAT:data+=struct.pack('<f',v)
        elif fd.type==F.TYPE_DOUBLE:data+=struct.pack('<d',v)
        else:
            n=int(v)&((1<<64)-1)
            while n>=128:data.append((n&127)|128);n>>=7
            data.append(n)
    return raw(bytes(data))

def single(fd,v):
    n=fd.number
    if fd.type==F.TYPE_MESSAGE:return f'db({n},'+proto(v)+')'
    if fd.type==F.TYPE_STRING:return f'db({n},'+json.dumps(v)+')'
    if fd.type==F.TYPE_BYTES:return f'db({n},'+raw(v)+')'
    if fd.type==F.TYPE_FLOAT:return f'df({n},{int.from_bytes(struct.pack("<f",v),"little")}ULL,4)'
    if fd.type==F.TYPE_DOUBLE:return f'df({n},{int.from_bytes(struct.pack("<d",v),"little")}ULL,8)'
    return f'dv({n},'+integer(v)+')'

def proto(message):
    parts=[]
    for fd,v in message.ListFields():
        if fd.is_repeated:
            if fd.is_packed:parts.append(f'db({fd.number},'+packed(fd,v)+')')
            else:parts.extend(single(fd,x) for x in v)
        else:parts.append(single(fd,v))
    return '+'.join(parts) or 'std::string{}'

def emit(name,model):
    lines=['DART_GRAPH_BUILD static std::string '+name+'_graph() {','    std::string graph;']
    for fd,v in model.graph.ListFields():
        if fd.number==1:
            for node in v:
                assert {d.name for d,_ in node.ListFields()}<={'input','output','name','op_type','attribute'}
                inputs='{'+','.join(json.dumps(x) for x in node.input)+'}'
                outputs='{'+','.join(json.dumps(x) for x in node.output)+'}'
                attrs='{'+','.join(proto(a) for a in node.attribute)+'}'
                lines.append('    graph+=dn('+','.join([json.dumps(node.op_type),json.dumps(node.name),inputs,outputs,attrs])+');')
        elif fd.is_repeated:
            lines.extend('    graph+='+single(fd,x)+';' for x in v)
        else:lines.append('    graph+='+single(fd,v)+';')
    fields=[]
    for fd,v in model.ListFields():
        if fd.name=='graph':fields.append(f'db({fd.number},graph)')
        elif fd.is_repeated:fields.extend(single(fd,x) for x in v)
        else:fields.append(single(fd,v))
    lines+=['    return '+'+'.join(fields)+';','}\n']
    return '\n'.join(lines)
