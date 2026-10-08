"""Extract complete inner sequence zips (machine_hall/<SEQ>/<SEQ>.zip) from a partially downloaded outer zip.
Entries use data descriptors, so sizes are unknown in local headers: inflate the raw deflate stream until its end."""
import mmap, re, struct, zlib, sys, os, zipfile, subprocess
src = 'machine_hall.zip'; want = sys.argv[1:]
f = open(src, 'rb'); m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
for mt in re.finditer(rb'PK\x03\x04', m):
    p = mt.start(); h = m[p:p + 30]
    if len(h) < 30: break
    meth = struct.unpack('<H', h[8:10])[0]; fl, el = struct.unpack('<HH', h[26:30])
    name = m[p + 30:p + 30 + fl].decode(errors='replace')
    if not (name.endswith('.zip') and name.startswith('machine_hall/')): continue
    seq = name.split('/')[1]
    if want and seq not in want: continue
    if os.path.exists(f'{seq}/mav0/DONE'): print(seq, 'already'); continue
    d = zlib.decompressobj(-15); pos = p + 30 + fl + el; out = open(f'{seq}.zip', 'wb'); ok = False
    while pos < len(m):
        chunk = m[pos:pos + (1 << 24)]; pos += len(chunk)
        out.write(d.decompress(chunk))
        if d.eof: ok = True; break
    out.close()
    if not ok: print(seq, 'incomplete (stream not finished)'); os.remove(f'{seq}.zip'); continue
    zipfile.ZipFile(f'{seq}.zip').extractall(seq); os.remove(f'{seq}.zip')
    open(f'{seq}/mav0/DONE', 'w').write('ok'); print(seq, 'extracted', sorted(os.listdir(f'{seq}/mav0')))
