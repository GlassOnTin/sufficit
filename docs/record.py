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
            "obstacles": SPH_OBSTACLES, "J": J, "wall_s": wall,
            "device": "cpu"}


def sph_scatter():
    """How reproducible is the delivered impulse at a FIXED resolution?

    The GPU engine scatters forces with atomics, so every run perturbs
    the summation order by about one ulp and changes nothing else.
    Repeated runs are therefore independent draws of the same model
    under rounding, which is the cleanest probe available of how much
    of a ladder difference is signal. The CPU value belongs in the
    sample as one more draw: it is bitwise reproducible, and that only
    means it always returns the same draw.

    Needs cupy. Four at a time, because the engine is launch-bound at
    these sizes and the device is not close to full."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    # Two functionals, because the page runs two. The cfg rungs are
    # sph_wall_impulse, integrated to T=3.2, which is what the recorded
    # funnels use. The "live" rungs are the bare wall integrated to
    # T=4.6, which is what the page's own headline ladder uses, and a
    # floor measured on one functional says nothing about the other.
    jobs = [(cfg, n, k) for cfg, n in (("plain", 64), ("plain", 96),
                                       ("plain", 144), ("low", 144),
                                       ("low", 192), ("live", 16),
                                       ("live", 24), ("live", 36),
                                       ("live", 48))
            for k in range(6)]
    draws = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as ex:
        for cfg, n, _k, J in ex.map(_sph_gpu_one, jobs):
            draws.setdefault(f"{cfg}/{n}", []).append(J)
            print(f"{cfg:5s} nres={n:3d}  J={J:.6f}", flush=True)
    return {"functional": "sph_wall_impulse(nres, obstacle, T=3.2)",
            "obstacles": SPH_OBSTACLES, "draws": draws, "device": "gpu",
            "note": "each draw is one ulp-level perturbation of the "
                    "summation order and nothing else; the CPU value "
                    "for the same rung lives in sph_funnel.json and "
                    "counts as one more draw"}


def _sph_gpu_one(job):
    cfg, n, k = job
    sys.path.insert(0, ROOT)
    import numpy
    import sufficit as sf
    if cfg == "live":       # the page's own functional: bare wall, T=4.6
        o = sf.sph_dam_break(nres=n, device="gpu")
        J = float(numpy.sum(o["F"]) * (o["ts"][1] - o["ts"][0]))
    else:
        J = sf.sph_wall_impulse(n, obstacle=SPH_OBSTACLES[cfg],
                                device="gpu")
    return cfg, n, k, J


GPU_NRES = (64, 96, 144, 216, 324)


def sph_gpu_funnel():
    """The impulse ladder the CPU engine could not reach, on the GPU
    engine and at a fixed refinement ratio of 1.5 throughout, so it
    holds three overlapping triples: (64, 96, 144) checked at 216,
    (96, 144, 216) checked at 324, and (144, 216, 324) with nothing
    finer to check it. Measured on the recorded snapshots (see
    sph_crest, which replaced an asserted thickness with this): the
    fluid in a 0.2-wide strip past the crest is 62 particle spacings
    deep at 324 against 3.5 at nres=27, the finest rung the page's own
    berm ladder runs. Every rung is GPU, because a ladder must not mix
    devices.
    Needs cupy; about an hour on an RTX 5090."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    jobs = sorted(((c, n, 0) for c in SPH_OBSTACLES for n in GPU_NRES),
                  key=lambda j: -j[1])
    J = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as ex:
        for cfg, n, _k, val in ex.map(_sph_gpu_one, jobs):
            J.setdefault(cfg, {})[str(n)] = val
            print(f"{cfg:5s} nres={n:3d}  J={val:.6f}", flush=True)
    return {"functional": "sph_wall_impulse(nres, obstacle, T=3.2)",
            "obstacles": SPH_OBSTACLES, "J": J, "device": "gpu",
            "ratio": 1.5}


CREST_BOX = ((3.05, 3.70), (0.0, 0.50))


def sph_crest():
    """The crest jet at nres=324, cropped so a page can draw it.

    The page argues that the 12% berm refuses because its jet is
    under-resolved, and until now it asserted a thickness rather than
    measuring one. This records the particles themselves. Only the
    crop is kept: 9,317 of 104,976, because 105,000 SVG circles is 7 MB
    and the argument lives at the crest.

    Also records the count in a 0.2-wide window past the crest at any
    height, which is the resolution-robust way to say how well the
    sheet is resolved. A single vertical cut is not: the jet wanders,
    so one cut catches spray at one resolution and a sheet at another.
    Volume is well defined where a pointwise thickness is not."""
    import numpy
    sys.path.insert(0, ROOT)
    import sufficit as sf
    o = sf.sph_dam_break(nres=324, T=3.6, snapshots=(2.8,),
                         obstacle=(2.9, 0.4, 0.12), device="gpu")
    t, x, y, p = o["snaps"][0]
    (a, b), (c, d) = CREST_BOX
    m = (x >= a) & (x <= b) & (y >= c) & (y <= d)
    pm = float(p[m].max())
    return {"nres": 324, "device": "gpu", "t": round(float(t), 3),
            "obstacle": [2.9, 0.4, 0.12],
            "box": [list(CREST_BOX[0]), list(CREST_BOX[1])],
            "dx": 1.0 / 324,
            "x": [round(float(v), 3) for v in x[m]],
            "y": [round(float(v), 3) for v in y[m]],
            "p": [round(float(v) / pm, 2) for v in p[m]],
            "window": {"x0": 3.30, "x1": 3.50,
                       "count": int(((x >= 3.30) & (x < 3.50)).sum())}}


RECIPES = {"h10_ladder": h10_ladder, "sph_funnel": sph_funnel,
           "sph_scatter": sph_scatter, "sph_gpu_funnel": sph_gpu_funnel,
           "sph_crest": sph_crest}


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
