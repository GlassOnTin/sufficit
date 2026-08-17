"""Record expensive runs for the case pages to restore.

Every page regenerates every certificate it shows. What some pages
cannot afford is the run that produces the certificate's inputs: hours
of molecular dynamics, a window diagonalization with a 36 GB working
set, fifty-minute SPH ladders. Those runs happen here, on a machine
that can hold them, and land in data/<slug>.json with the provenance
needed to audit them later: when, on what machine and stack, against
which library source. At build time the page restores the recorded
measurements, recomputes the cheap certification from them, and states
in its footer which half is live and which is not.

The provenance carries a hash of sufficit.py as it was when the run
happened. The pages compare it against the current file and, when they
differ, say so in the footer: a recorded number is a claim about the
library that produced it, and "the library has changed since" is a
fact the reader is owed. Re-record to clear it.

Usage: python3 docs/record.py <slug> [<slug> ...]
Slugs: h10_ladder, sph_funnel. Overwrites data/<slug>.json.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def library_sha():
    with open(os.path.join(ROOT, "sufficit.py"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def provenance():
    import numpy
    import scipy
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    return {"recorded": time.strftime("%Y-%m-%d"),
            "box": f"{os.cpu_count()}-core {platform.machine()}",
            "stack": f"numpy {numpy.__version__} / scipy {scipy.__version__}",
            "commit": commit,
            "library_sha": library_sha()}


def h10_ladder():
    """The H10 window ladder past what a page build can afford. ell=7
    alone is half an hour; ell=8 holds a 36 GB working set. The odd
    ell=7 tiles as [7,3] and its blocks carry a degenerate ground
    doublet, so that rung's value can move between numpy/scipy stacks;
    the even rungs cannot. The caveat travels with the record."""
    import sufficit as sf
    ells = {}
    for ell in (5, 6, 7, 8):
        t0 = time.perf_counter()
        c = sf.h_chain_bracket(10, 1.8, ell=ell)
        ells[str(ell)] = {"value": c.value, "err": c.err,
                          "wall_s": round(time.perf_counter() - t0, 1)}
    return {"call": "h_chain_bracket(10, 1.8, ell)", "ells": ells,
            "note": "ell=7 tiles [7,3]; odd blocks carry a degenerate "
                    "doublet, so its value is stack-dependent. Even "
                    "rungs are not."}


SPH_OBSTACLES = {"plain": None,
                 "tall": (2.9, 0.4, 0.4),
                 "low": (2.9, 0.4, 0.12)}
SPH_NRES = (16, 24, 32, 36, 48, 64, 72, 96, 144, 192)


def _sph_one(job):
    cfg, n = job
    import sufficit as sf
    t0 = time.perf_counter()
    J = sf.sph_wall_impulse(n, obstacle=SPH_OBSTACLES[cfg])
    return cfg, n, J, round(time.perf_counter() - t0, 1)


def sph_funnel():
    """The sea-wall impulse at resolutions the page build cannot
    afford: the finest rungs are 36,864 fluid particles and near an
    hour each. Runs are independent, so they fan out over processes."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = sorted(((c, n) for c in SPH_OBSTACLES for n in SPH_NRES),
                  key=lambda j: -j[1])
    J, wall = {}, {}
    workers = max(2, (os.cpu_count() or 4) // 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for cfg, n, val, dt in ex.map(_sph_one, jobs):
            J.setdefault(cfg, {})[str(n)] = val
            wall.setdefault(cfg, {})[str(n)] = dt
            print(f"{cfg:5s} nres={n:3d}  J={val:.6f}  {dt:.0f}s",
                  flush=True)
    return {"functional": "sph_wall_impulse(nres, obstacle, T=3.2)",
            "obstacles": SPH_OBSTACLES, "J": J, "wall_s": wall}


RECIPES = {"h10_ladder": h10_ladder, "sph_funnel": sph_funnel}


def write(slug, payload):
    payload = dict(payload, slug=slug, provenance=provenance())
    path = os.path.join(ROOT, "data", f"{slug}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    slugs = sys.argv[1:]
    if not slugs or any(s not in RECIPES for s in slugs):
        sys.exit(f"usage: record.py <slug>; slugs: {', '.join(RECIPES)}")
    for slug in slugs:
        write(slug, RECIPES[slug]())
