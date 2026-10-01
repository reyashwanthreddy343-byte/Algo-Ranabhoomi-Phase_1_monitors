"""Monitor Placement solver: greedy -> exact swap local search -> iterated local search.

Pipeline
1. Greedy add (vectorised) -> submit immediately so checkpoint 1 (5% of budget) is covered.
2. Fast-interchange local search: the FULL (remove r, add s) neighbourhood is evaluated
   in one numpy shot using each point's best / second-best open-site cost.
3. Iterated local search: perturb a few open sites, re-descend, keep if not worse,
   restart from the best occasionally. Every strict improvement is submitted.

If numpy is missing, a pure-Python greedy + swap fallback is used.
"""

import random
import time

from adapter import Solver, dist_table

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

SAFETY_S = 0.45  # stop searching when this much budget is left


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        t0 = time.perf_counter()
        if np is None:
            return self._fallback(instance, submit_candidate)

        T = np.asarray(dist_table(instance), dtype=np.int64)  # n x m
        W = np.asarray(instance.weights, dtype=np.int64)       # n
        n, m = T.shape
        k = instance.k
        WT = T * W[:, None]  # weighted table
        rng = random.Random(12345)

        state = {"best": None, "best_cost": None, "remaining": None}

        def submit(sites, cost):
            if state["best_cost"] is not None and cost >= state["best_cost"]:
                return
            state["best_cost"] = cost
            state["best"] = list(sites)
            rec = submit_candidate({"sites": [int(s) for s in sites]})
            state["remaining"] = rec.get("remaining_s", None)

        def out_of_time():
            r = state["remaining"]
            if r is not None:
                # remaining_s is as of the last receipt; also guard with local clock
                return (r - (time.perf_counter() - state["last_t"])) < SAFETY_S
            return (time.perf_counter() - t0) > 4.0

        state["last_t"] = t0

        def total(sites):
            return int((W * T[:, sites].min(axis=1)).sum())

        # ---- greedy ----
        cur = np.full(n, instance.penalty, dtype=np.int64)
        opened = []
        mask = np.zeros(m, dtype=bool)
        for _ in range(k):
            cost_if = (W[:, None] * np.minimum(T, cur[:, None])).sum(axis=0)
            cost_if[mask] = np.iinfo(np.int64).max
            s = int(cost_if.argmin())
            opened.append(s)
            mask[s] = True
            cur = np.minimum(cur, T[:, s])
        c = total(opened)
        submit(opened, c)
        state["last_t"] = time.perf_counter()

        def descend(sites):
            """Best-improvement swaps until local optimum. Returns (sites, cost)."""
            sites = list(sites)
            while True:
                sub = T[:, sites]                       # n x k
                order = np.argsort(sub, axis=1)
                a1 = order[:, 0]
                rows = np.arange(n)
                d1 = sub[rows, a1]
                if k > 1:
                    d2 = sub[rows, order[:, 1]]
                else:
                    d2 = np.full(n, instance.penalty, dtype=np.int64)
                m1 = np.minimum(T, d1[:, None])                          # n x m
                m2 = np.minimum(T, d2[:, None])
                A = (W[:, None] * (m1 - d1[:, None])).sum(axis=0)        # m  (<= 0)
                diff = W[:, None] * (m2 - m1)                            # n x m
                B = np.zeros((k, m), dtype=np.int64)
                np.add.at(B, a1, diff)                                   # rows by assigned site
                delta = A[None, :] + B                                   # k x m
                delta[:, sites] = np.iinfo(np.int64).max // 4            # no re-adding open sites
                idx = int(delta.argmin())
                r, s = divmod(idx, m)
                if delta[r, s] >= 0:
                    return sites, total(sites)
                sites[r] = int(s)

        sites, c = descend(opened)
        submit(sites, c)
        state["last_t"] = time.perf_counter()

        # ---- iterated local search ----
        cur_sites, cur_cost = list(sites), c
        best_sites, best_cost = list(sites), c
        stall = 0
        while not out_of_time():
            cand = list(cur_sites)
            kick = rng.choice((1, 2, 2, 3, 4)) if k > 4 else 1
            kick = min(kick, k)
            in_set = set(cand)
            for pos in rng.sample(range(k), kick):
                for _ in range(20):
                    s = rng.randrange(m)
                    if s not in in_set:
                        in_set.discard(cand[pos])
                        cand[pos] = s
                        in_set.add(s)
                        break
            cand, cc = descend(cand)
            if cc <= cur_cost:
                cur_sites, cur_cost = cand, cc
            if cc < best_cost:
                best_sites, best_cost = list(cand), cc
                stall = 0
                submit(best_sites, best_cost)
                state["last_t"] = time.perf_counter()
            else:
                stall += 1
                if stall % 40 == 0:
                    cur_sites, cur_cost = list(best_sites), best_cost

        return {"sites": [int(s) for s in state["best"]]}

    # ------------------------------------------------------------------ fallback
    def _fallback(self, instance, submit_candidate):
        table = dist_table(instance)
        weights, m, k = instance.weights, len(instance.sites), instance.k

        def cost_with(base, s):
            return sum(w * min(b, row[s]) for w, b, row in zip(weights, base, table))

        opened, current = [], [instance.penalty] * instance.size
        for _ in range(k):
            s = min((s for s in range(m) if s not in opened), key=lambda s: cost_with(current, s))
            opened.append(s)
            current = [min(b, row[s]) for b, row in zip(current, table)]
        receipt = submit_candidate({"sites": list(opened)})
        best = receipt["cost"]
        improved = True
        while improved and receipt["remaining_s"] > 0.5:
            improved = False
            for pos in range(k):
                if receipt["remaining_s"] < 0.5:
                    break
                rest = opened[:pos] + opened[pos + 1:]
                without = [min((row[s] for s in rest), default=instance.penalty) for row in table]
                for s in range(m):
                    if s in opened:
                        continue
                    c = cost_with(without, s)
                    if c < best:
                        best, opened[pos], improved = c, s, True
                        without = [min((row[x] for x in opened[:pos] + opened[pos + 1:]), default=instance.penalty)
                                   for row in table]
                receipt = submit_candidate({"sites": list(opened)})
        return {"sites": opened}
