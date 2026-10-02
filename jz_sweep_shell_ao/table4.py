# Table 4 numbers: loop wall time, solver time (setup timer + summed solve-phase times), rest = loop - solver.
# Medians over reps. Usage: python3 table4.py <sweep dir>
import re, glob, statistics as st, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
def parse(f):
    t = open(f).read()
    wall = float(re.findall(r"Wall time \(loop\)\s+:\s+([0-9.]+)", t)[0])
    setup = float(re.findall(r"timer analyze\+factorize:\s+([0-9.eE+-]+) s", t)[-1])
    sol = sum(float(x) for x in re.findall(r"(?:CuDSS|PARDISO) Solve Phase\.\.\. Done in ([0-9.eE+-]+) ms", t)) / 1e3
    anl = len(re.findall(r"Analysis Phase", t))
    tip = re.findall(r"Final avg tip pos\s+:\s+\((.+)\) m", t)[0]
    res = max(float(x) for x in re.findall(r"\|residual\| = ([0-9.eE+-]+)", t))
    return dict(wall=wall, solver=setup + sol, rest=wall - setup - sol, anl=anl, tip=tip, res=res)
print(f"{'grid':>4} | {'loopP':>6} {'loopC':>6} {'x':>5} | {'solvP':>6} {'solvC':>6} {'x':>5} | {'restP':>6} {'restC':>6} | anl P/C | tipsame | maxres")
for g in (10, 20, 30, 40, 50, 70, 100, 140, 200):
    m = {}
    for s in ("pardiso", "cudss"):
        R = [parse(f) for f in sorted(glob.glob(f"{d}/g{g}_{s}_*.log"))]
        if not R: break
        m[s] = {k: st.median(r[k] for r in R) for k in ("wall", "solver", "rest")}
        m[s].update(anl=R[0]["anl"], tips={r["tip"] for r in R}, res=max(r["res"] for r in R))
    if len(m) < 2: continue
    P, C = m["pardiso"], m["cudss"]
    print(f"{g:>4} | {P['wall']:6.2f} {C['wall']:6.2f} {P['wall']/C['wall']:5.2f} | {P['solver']:6.2f} {C['solver']:6.2f} {P['solver']/C['solver']:5.2f} | {P['rest']:6.2f} {C['rest']:6.2f} | {P['anl']:>3}/{C['anl']:<3} | {P['tips']==C['tips'] and len(P['tips'])==1} | {max(P['res'],C['res']):.0e}")
