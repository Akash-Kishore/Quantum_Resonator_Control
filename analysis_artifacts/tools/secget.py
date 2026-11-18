import re, sys
f, pat = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 120
L = open(f, encoding="utf-8", errors="replace").read().splitlines()
H = re.compile(r"^(STAGE |SECTION |APPENDIX )")
if pat == "--list":
    for i, l in enumerate(L, 1):
        if H.match(l): print(i, l[:100])
    sys.exit(0)
rx = re.compile(pat)
for i, l in enumerate(L):
    if H.match(l) and rx.search(l):
        j = i + 1
        while j < len(L) and not H.match(L[j]): j += 1
        out = L[i:j]
        print("\n".join(out[:mx]))
        if len(out) > mx: print("... [%d more lines; rerun with a larger maxlines]" % (len(out) - mx))
        break
else:
    print("NOT FOUND")
