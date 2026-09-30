import re, statistics as st, glob
def parse(f):
    t=open(f).read(); ph={}
    for name,ms in re.findall(r"(?:CuDSS|PARDISO) (\w+) Phase\.\.\. Done in ([0-9.eE+-]+) ms",t):
        ph.setdefault(name,[]).append(float(ms))
    g=lambda p: re.findall(p,t)
    return dict(
        n=int(g(r"n = (\d+)")[0]), nnz=int(g(r"nnz = (\d+)")[0]),
        wall=float(g(r"Wall time \(loop\)\s+:\s+([0-9.]+)")[0]),
        setup=float(g(r"timer analyze\+factorize:\s+([0-9.eE+-]+) s")[-1]),
        asm=float(g(r"timer assembly matrix:\s+([0-9.eE+-]+) s")[-1]),
        anl=st.median(ph["Analysis"]), fac=st.median(ph["Factorization"]), sol=st.median(ph["Solve"]),
        calls=len(ph["Analysis"]), res=max(float(x) for x in g(r"\|residual\| = ([0-9.eE+-]+)")),
        tip=g(r"Final avg tip pos\s+:\s+\((.+)\) m")[0].replace(",","").split(),
        rss=int(g(r"peakRSS=(\d+)kB")[0])/1e6)
rows={}
for res in (0,2,4,8):
    for s in ("pardiso","cudss"):
        R=[parse(f) for f in sorted(glob.glob(f"res{res}_{s}_*.log"))]
        m={k:st.median([r[k] for r in R]) for k in ("wall","setup","asm","anl","fac","sol")}
        m.update(n=R[0]["n"],nnz=R[0]["nnz"],calls=R[0]["calls"],res=max(r["res"] for r in R),
                 tip=R[0]["tip"], tips={tuple(r["tip"]) for r in R}, rss=R[0]["rss"],
                 wmin=min(r["wall"] for r in R), wmax=max(r["wall"] for r in R))
        rows[(res,s)]=m
print(f"{'res':>4} {'n':>9} {'nnz':>11} | {'wall P':>7} {'wall C':>7} {'x':>5} | {'anl P':>7} {'anl C':>7} {'x':>5} | {'fac P':>6} {'fac C':>6} {'x':>5} | {'sol P':>6} {'sol C':>6} {'x':>5} | {'fac+sol x':>9} | {'setupP':>7} {'setupC':>7} | {'asmP':>6} {'asmC':>6}")
for res in (0,2,4,8):
    P,C=rows[(res,"pardiso")],rows[(res,"cudss")]
    print(f"{res:>4} {P['n']:>9,} {P['nnz']:>11,} | {P['wall']:7.2f} {C['wall']:7.2f} {P['wall']/C['wall']:5.2f} | {P['anl']:7.1f} {C['anl']:7.1f} {P['anl']/C['anl']:5.2f} | {P['fac']:6.2f} {C['fac']:6.2f} {P['fac']/C['fac']:5.1f} | {P['sol']:6.2f} {C['sol']:6.2f} {P['sol']/C['sol']:5.0f} | {(P['fac']+P['sol'])/(C['fac']+C['sol']):9.1f} | {P['setup']:7.2f} {C['setup']:7.2f} | {P['asm']:6.2f} {C['asm']:6.2f}")
print("\nchecks:")
for res in (0,2,4,8):
    P,C=rows[(res,"pardiso")],rows[(res,"cudss")]
    print(f" res{res}: calls={P['calls']}/{C['calls']}  wall spread P {P['wmin']:.2f}-{P['wmax']:.2f}  C {C['wmin']:.2f}-{C['wmax']:.2f}  maxres P {P['res']:.0e} C {C['res']:.0e}  tipsP==tipsC: {P['tips']==C['tips']} ({len(P['tips'])} distinct)  tip={' '.join(P['tip'])}  RSS P {P['rss']:.1f}GB C {C['rss']:.1f}GB")
