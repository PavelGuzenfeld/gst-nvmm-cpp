import re, statistics as st
rows=[]
for line in open("timing.log"):
    m=dict(re.findall(r"(\w+)=([0-9.]+)", line))
    if "memenc_gpu" in m: rows.append({k:float(v) for k,v in m.items()})

def report(rows, label):
    print(f"\n===== {label}  (n={len(rows)}) =====")
    def stat(k):
        xs=sorted(r[k] for r in rows)
        return f"median={st.median(xs):8.3f}  mean={st.mean(xs):8.3f}  p10={xs[len(xs)//10]:8.3f}  p90={xs[9*len(xs)//10]:8.3f}"
    for k in ["enc","prebox","tail","memenc_gpu","tail_bubble"]:
        print(f"  {k:12s} ms : {stat(k)}")
    fr=[r["enc"]+r["prebox"]+r["tail"] for r in rows]
    print(f"  frame_total  ms : median={st.median(fr):8.3f}  mean={st.mean(fr):8.3f}")
    tf=sorted(100*r["tail"]/(r["enc"]+r["prebox"]+r["tail"]) for r in rows)
    bf=sorted(100*r["tail_bubble"]/(r["enc"]+r["prebox"]+r["tail"]) for r in rows)
    print(f"  tail/frame    % : median={st.median(tf):7.2f}  mean={st.mean(tf):7.2f}  p90={tf[9*len(tf)//10]:7.2f}")
    print(f"  bubble/frame  % : median={st.median(bf):7.2f}  mean={st.mean(bf):7.2f}  p90={bf[9*len(bf)//10]:7.2f}")

report(rows, "ALL frames")
report(rows[50:], "STEADY-STATE (frame 50+)")
