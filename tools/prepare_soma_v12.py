#!/usr/bin/env python3
"""Development-only, standard-library preparation of the pinned commercial v1.2 GGUF.
Normal Tide Motion generation is native and never invokes this tool or Python.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source, output, download, accept):
    if not accept:
        raise ValueError('Read licenses/NVIDIA-KIMODO-MODEL.txt and pass --accept-licenses')
    pins = json.loads(Path(__file__).with_name('soma-rp-v1.2.json').read_text())
    if output.exists():
        raise ValueError('Output already exists')
    for name, pin in pins['files'].items():
        path = source / name
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.part')
            url = f"https://huggingface.co/{pins['model']}/resolve/{pins['revision']}/{name}"
            print(f'Downloading {name}', flush=True)
            subprocess.run(['curl', '--fail', '--location', '--proto', '=https',
                            '--proto-redir', '=https', '--retry', '3', '--max-time', '7200',
                            '--output', str(temporary), url], check=True)
            if temporary.stat().st_size != pin['bytes'] or digest(temporary) != pin['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
            temporary.replace(path)
        if not path.is_file() or path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
            raise ValueError(f'Missing or checksum mismatch: {name}')
    (source / 'REVISION').write_text(f"{pins['revision']} {pins['model']}\n")
    subprocess.run([sys.executable, str(Path(__file__).with_name('convert_motion_to_gguf.py')),
                    '--input', str(source), '--output', str(output)], check=True)
    if digest(output) != pins['ggufSha256']:
        output.unlink()
        raise ValueError('Converted GGUF checksum mismatch')
    print(f'Prepared verified {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--download', action='store_true')
    p.add_argument('--accept-licenses', action='store_true')
    args = p.parse_args()
    try:
        prepare(args.input, args.output, args.download, args.accept_licenses)
    except (ValueError, OSError) as error:
        p.exit(1, str(error) + '\n')
