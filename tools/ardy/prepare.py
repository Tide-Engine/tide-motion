"""Developer-only pinned commercial checkpoint downloader; no evaluation data."""
import argparse
import hashlib
import json
import pathlib
import urllib.request

REVISION='e6c6804fb6b55636ad927145d74959fd46882ccd'
BASE='https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60/resolve/'+REVISION+'/'
FILES=json.loads(pathlib.Path(__file__).with_name('checkpoint-files.json').read_text(encoding='utf-8-sig'))

def verify(root):
    for name,spec in FILES.items():
        path=root/name
        if path.stat().st_size!=spec['bytes'] or hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()!=spec['sha256']:
            raise RuntimeError('Pinned checkpoint mismatch: '+name)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('destination',type=pathlib.Path)
    p.add_argument('--accept-licenses',action='store_true',required=True,
                   help='Accept the pinned bundled NVIDIA Open Model License (2025), see notes/ardy-native.md')
    a=p.parse_args()
    for name,spec in sorted(FILES.items(),key=lambda kv:kv[0]!='LICENSE'):
        path=a.destination/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==spec['bytes'] and hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()==spec['sha256']:
            continue
        print('Downloading',name,flush=True)
        temp=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(BASE+name,timeout=180) as response,open(temp,'wb') as out:
            if not response.url.startswith('https://'):raise RuntimeError('HTTPS required')
            total=0
            while chunk:=response.read(1024*1024):
                total+=len(chunk)
                if total>spec['bytes']:raise RuntimeError('Oversized download')
                out.write(chunk)
        if temp.stat().st_size!=spec['bytes'] or hashlib.file_digest(open(temp,'rb'),'sha256').hexdigest()!=spec['sha256']:
            raise RuntimeError('Download digest mismatch: '+name)
        temp.replace(path)
    verify(a.destination)

if __name__=='__main__':main()
