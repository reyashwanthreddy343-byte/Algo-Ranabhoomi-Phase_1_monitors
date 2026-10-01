"""Windows-friendly local test: runs MySolver directly (no child process).
Usage:  python test_local.py
"""
import json
import time

import data
from adapters.mine import MySolver

ref = json.load(open("public_reference.json"))["instances"]
BUDGET = 5.0
all_ok = True
print(f"{'instance':12} {'cost':>10} {'reference':>10}  result")
for e in data.PUBLIC_SUITE:
    inst = data.make_instance(e.seed, e.profile, e.name)
    table = data.effective_table(inst)
    t0 = time.perf_counter()
    state = {"best": None}

    def cost(plan):
        s = plan["sites"]
        assert len(s) == inst.k and len(set(s)) == inst.k, "need exactly k distinct sites"
        assert all(0 <= x < len(inst.sites) for x in s), "site id out of range"
        return sum(w * min(row[x] for x in s) for w, row in zip(inst.weights, table))

    def submit(plan):
        el = time.perf_counter() - t0
        c = cost(plan)
        if state["best"] is None or c < state["best"]:
            state["best"] = c
        return {"accepted": True, "reason": "", "cost": c, "best": state["best"],
                "elapsed_s": el, "remaining_s": BUDGET - el}

    final = MySolver().solve(inst, submit)
    submit(final)
    took = time.perf_counter() - t0
    r = ref[e.name]["reference_cost"]
    ok = state["best"] <= r and took < BUDGET * 1.05
    all_ok &= ok
    print(f"{e.name:12} {state['best']:>10} {r:>10}  {'OK' if ok else 'WORSE/SLOW'}  ({took:.1f}s)")
print("ALL GOOD" if all_ok else "SOMETHING IS OFF")
