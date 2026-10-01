# Algo Ranabhoomi - Phase 1

Problem chosen: Monitor Placement (monitors/)

Solution code: monitors/adapters/mine.py
Explanation: monitors/SOLUTION.md

Approach: greedy start, then swap local search over all (remove, add)
site pairs, then iterated local search. Reaches the reference cost on
all 6 public instances (100/100).

Run:
cd monitors
python run.py --adapter adapters.mine:MySolver --out report.json
