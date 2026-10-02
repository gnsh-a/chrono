# Memcpy cost per call from the instrumented cuDSS runs (memcpy/<cfg>/g<grid>_cudss_1.log).
import re, statistics as st, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def ph(t, name): return [float(x) for x in re.findall(rf"CuDSS {name} Phase\.\.\. Done in ([0-9.eE+-]+) ms", t)]
print(f"{'grid':>4} {'cfg':>4} {'n':>7} {'nnz':>10} {'calls':>5} | {'mat ms/call':>11} {'MB':>6} {'GB/s':>5} | {'rhs+sol ms':>10} | {'total s':>7} {'% wall':>6}")
for g in (10, 20, 30, 40, 50, 70, 100, 140, 200):
    for cfg in ("cold", "ao"):
        t = open(f"memcpy/{cfg}/g{g}_cudss_1.log").read()
        n, nnz = map(int, re.findall(r"n = (\d+)\s+nnz = (\d+)", t)[0])
        mat, rhs, sol = ph(t, "CopyMatrixH2D"), ph(t, "CopyRhsH2D"), ph(t, "CopySolD2H")
        wall = float(re.findall(r"Wall time \(loop\)\s+:\s+([0-9.]+)", t)[0])
        steady = mat[1:] if cfg == "ao" else mat          # analyze-once: first call also sends indices
        mb = (nnz * 8 + (0 if cfg == "ao" else nnz * 4 + (n + 1) * 4)) / 1e6
        m = st.median(steady); vec = st.median(rhs) + st.median(sol)
        tot = (sum(mat) + sum(rhs) + sum(sol)) / 1e3
        print(f"{g:>4} {cfg:>4} {n:>7} {nnz:>10} {len(mat):>5} | {m:11.2f} {mb:6.1f} {mb/m:5.1f} | {vec:10.3f} | {tot:7.3f} {100*tot/wall:6.2f}")
