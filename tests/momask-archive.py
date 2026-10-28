"""Inspect ONLY official ZIP directory and opt.txt metadata using bounded ranges.

No checkpoint, normalization arrays, motion data, or body model is extracted.
Run with stdlib Python: python tests/momask-archive.py IGNORED_OUTPUT_DIRECTORY
Fails on a changed archive directory or a server ignoring Range.
"""
import hashlib
from html.parser import HTMLParser
import json
import pathlib
import struct
import subprocess
import sys
import urllib.parse
import zlib

inventory = json.loads((pathlib.Path(__file__).parents[1] / 'notes/momask-archive-inventory.json').read_text())
base = 'https://drive.usercontent.google.com/download'
query = {'id': '1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0', 'export': 'download'}
class Form(HTMLParser):
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'input' and attrs.get('type') == 'hidden' and attrs.get('name') in ('confirm', 'uuid'):
            query[attrs['name']] = attrs['value']

def curl(url, size, byte_range=None):
    args = ['curl.exe', '--fail', '--silent', '--show-error', '--location',
            '--proto', '=https', '--proto-redir', '=https', '--max-time', '30', '--max-filesize', str(size)]
    if byte_range:
        args += ['--range', byte_range]
    result = subprocess.run(args + [url], check=True, capture_output=True, timeout=35)
    assert len(result.stdout) <= size
    return result.stdout
html = curl(base + '?' + urllib.parse.urlencode(query), 16384)
Form().feed(html.decode())
assert query.get('confirm') == 't', 'Drive metadata form changed'
url = base + '?' + urllib.parse.urlencode(query)
def fetch(offset, size):
    assert 0 < size <= 8192
    data = curl(url, size, f'{offset}-{offset+size-1}')
    assert len(data) == size
    return data

directory = fetch(inventory['central_directory_offset'], inventory['central_directory_bytes'])
assert hashlib.sha256(directory).hexdigest() == inventory['central_directory_sha256'], 'Archive changed'
output = pathlib.Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=False)
configs = []
for entry in inventory['entries']:
    if not entry['name'].endswith('/opt.txt'):
        continue
    header = struct.unpack('<4s5H3L2H', fetch(entry['local_offset'], 30))
    assert header[0] == b'PK\x03\x04' and header[3] == entry['method'] == 8
    payload = entry['local_offset'] + 30 + header[-2] + header[-1]
    data = zlib.decompress(fetch(payload, entry['compressed']), -15)
    assert len(data) == entry['size'] and zlib.crc32(data) == header[6]
    text = data.decode('utf-8')
    configs.append({'name': entry['name'], 'sha256': hashlib.sha256(data).hexdigest(), 'text': text})
(output / 'configs.json').write_text(json.dumps(configs, indent=2))
print(json.dumps(configs, indent=2))
