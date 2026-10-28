"""Generate non-model Unicode/HTML source tables from CPython 3.11's standard library.
Unicode 14.0.0; Unicode data license and PSF license retained in licenses/.
No tokenizer vocabulary, BPE merge, checkpoint or motion data is emitted.
"""
import unicodedata as u,html.entities,pathlib
assert u.unidata_version=='14.0.0'
def literal(s):return '"'+''.join('\\x%02x'%v for v in s.encode('utf8'))+'"'
out=['// Generated standard Unicode 14.0.0 / CPython 3.11 HTML facts; not model data.\n']
ranges=[];begin=0;kind=0
for cp in range(0x110001):
 c=chr(cp) if cp<0x110000 else '\0';cat=u.category(c)
 k=1 if cat[0]=='L' else 2 if cat[0]=='N' else 3 if c.isspace() else 0
 if k!=kind:
  if kind:ranges.append((begin,cp-1,kind))
  begin=cp;kind=k
out+=['static constexpr Range unicode_ranges[]={'+','.join('{%d,%d,%d}'%r for r in ranges)+'};\n']
out+=['static constexpr Case unicode_lower[]={'+','.join('{%d,%s}'%(cp,literal(chr(cp).lower())) for cp in range(0x110000) if chr(cp).lower()!=chr(cp))+'};\n']
decomp=[];comb=[];comp=[]
for cp in range(0x110000):
 c=chr(cp);n=u.normalize('NFD',c)
 if n!=c and not 0xac00<=cp<0xd7a4:decomp.append((cp,[ord(v) for v in n]))
 if u.combining(c):comb.append((cp,u.combining(c)))
 d=u.decomposition(c)
 if d and not d.startswith('<'):
  vals=[int(v,16) for v in d.split()]
  if len(vals)==2 and u.normalize('NFC',''.join(map(chr,vals)))==c:comp.append((*vals,cp))
out+=['static constexpr Decomp unicode_decomp[]={'+','.join('{%d,{%s},%d}'%(cp,','.join(map(str,v)),len(v)) for cp,v in decomp)+'};\n']
out+=['static constexpr Combining unicode_combining[]={'+','.join('{%d,%d}'%v for v in comb)+'};\n']
out+=['static constexpr Composition unicode_composition[]={'+','.join('{%d,%d,%d}'%v for v in sorted(comp))+'};\n']
out+=['static constexpr Entity html_entities[]={'+','.join('{%s,%s}'%(literal(k),literal(v)) for k,v in sorted(html.entities.html5.items()))+'};\n']
pathlib.Path('src/generated/clip-unicode.inc').write_text(''.join(out))
