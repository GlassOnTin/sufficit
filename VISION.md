# Sufficit

**A query compiler for physics.** You declare the model, the question, and the error you can tolerate; the system searches for the compression, the algorithm, and the certificate.

> **Status:** design phase. This document is the founding vision — an invitation to argue with it. The name (from the Latin *sufficere*, "it suffices") is a working title; the central object of the project is the *minimal sufficient statistic of a physical question*.

## The problem

Most of the flops in scientific computing are spent resolving detail nobody queried. Simulation codes evolve full microstates, but the questions we actually ask — a phase shift, a susceptibility, a binding energy, a transport coefficient — are radically contractive functionals of those microstates. The sloppy-models literature made this quantitative: across fields, the singular values of the parameters-to-predictions map decay geometrically, so the answers we want live on a few stiff directions of a "hyperribbon," not in the exponentially large state space we pay to represent.

The landmark exceptions prove the size of the prize. The fast multipole method turned O(N²) gravity into O(N) by finding the sufficient statistic of the Newtonian far field. DMRG effectively solved one-dimensional quantum matter by finding the sufficient statistic of gapped ground states. Lüscher's method extracts real-time scattering from Euclidean finite-volume energy levels; on-shell unitarity methods collapsed astronomical numbers of Feynman diagrams into a handful of physical building blocks; the conformal bootstrap squeezed record-precision critical exponents out of consistency conditions alone.

Each of these is the same event: someone notices that the map from model to answer discards almost everything, and finds coordinates in which the contraction is manifest. The events arrive roughly once per field per decade, cost years of a brilliant person, and the resulting insight typically lives and dies inside one community's codes.

## The thesis

Treat that discovery process as a compilation problem. The input is a triple — **(model, query class, error tolerance)** — and the output is an **(algorithm, certificate)** pair: a concrete computational plan exploiting whatever structure the query induces, together with a bound on the error incurred. A query optimizer for physics: specification is separated from schedule, schedule space is searched mechanically, and the result ships with its guarantee.

The underlying claim is physical, not merely computational: reachable states and askable questions occupy a small, structured corner of the formal state space, and that structure is *detectable* mechanically (rank probes, gap estimates, convergence radii, light cones) and *exploitable* mechanically (rewrite rules with composable error bounds).

## Why now

Every component of this system already exists in isolation. Kernel-independent and black-box fast multipole methods (Ying–Biros–Zorin; Fong–Darve) automated the analytic half of FMM two decades ago: hand the machine any smooth kernel as a black box and hierarchical compression is discovered numerically, with butterfly factorizations extending the reach to oscillatory operators. Randomized numerical linear algebra provides cheap rank probes — a literal compression detector you can point at any operator block. FFTW's planner, Spiral, and Halide proved that separating specification from schedule and searching the schedule space beats human tuning on problems humans care about. AlphaTensor, FunSearch, and AlphaEvolve settled the question of whether automated search can find genuinely new algorithms — a 4×4 matrix-multiplication bound stood for 56 years until it fell to search. And LLM-plus-proof-assistant pipelines are collapsing the cost of verification, which has always been the true bottleneck.

What no one has built is the thing that lets these components snap together.

## The missing piece: an error-algebra IR

The novel artifact is an **intermediate representation in which every rewrite carries a composable error bound**, so that approximation error propagates through arbitrary program transformations the way derivatives propagate through automatic differentiation.

Rewrites include: truncate a multipole expansion; compress an operator block to numerical rank r; cluster-expand a Gibbs state; project dynamics onto slow variables; smear a spectral function to finite resolution; replace a kernel with a learned surrogate. Each carries a bound whose *provenance is physical*: spectral gaps, Lieb–Robinson velocities, cluster-expansion convergence radii, temperature, noise rates, resolution parameters.

Small-scale existence proofs exist. Herbie and Rosa are real compilers that carry certified floating-point error through rewrites; Taylor-model arithmetic does it rigorously for ODEs. Nobody has lifted this machinery from *rounding* error to *model-reduction* error. That lift is the whole ballgame. With the IR, the search components above compose into a compiler; without it, the project degenerates into another uncertified neural surrogate. **The certificate is the product.**

Guarantees come in declared tiers — **rigorous** (machine-checkable bound), **asymptotic** (valid in a stated limit), **empirical** (statistically validated) — carried in the type system the way memory-safety guarantees are, so that degradation is explicit rather than silent.

## Design principles

1. **Query-first accounting.** Cost should scale with the precision of the question, not the sharpness of the formalism. The Hansen–Lupo–Tantalo move — making the experiment's finite resolution part of the computational target — is the canonical example, and it generalizes.
2. **Certificates are deliverables.** An answer without a bound is a hypothesis, not a result. Verification effort is a first-class budget item, not an afterthought.
3. **Honest degradation.** When rigorous bounds are out of reach (they often will be), say so in the types. An empirical certificate honestly declared beats a rigorous one falsely implied.
4. **The flywheel.** Every discovered compression becomes a reusable, certified rewrite rule in a shared library. The system should compound the way LLVM did, instead of the field's current mode where each insight is rediscovered per community.
5. **Rediscovery as regression test.** Before the compiler is trusted on open problems, it must mechanically recover the classics — FMM, hierarchical low-rank solvers, cluster expansions — from black-box inputs. Known answers are the test suite.

## Roadmap

Staged by certificate difficulty, easiest first.

**Phase 0 — the IR.** Define the error algebra. Wrap existing certified primitives (interval and Taylor-model arithmetic, randomized rank probes with failure probabilities, standard analytic bounds) as composable nodes. Deliverable: toy pipelines where end-to-end bounds survive nontrivial rewrite chains.

**Phase 1 — N-body with arbitrary kernels.** The compiler must rediscover FMM given only a black-box smooth kernel, then handle oscillatory (Helmholtz-class) kernels via directional and butterfly rewrites. Pure sanity check: the answers are known, the bounds are classical, failure is unambiguous.

**Phase 2 — Gibbs-state observables.** Automated high-temperature cluster expansions for local observables of lattice Hamiltonians. Convergence radii yield rigorous, temperature-dependent certificates, and the recent polynomial-time Gibbs-sampling results mark out the tractable region in advance.

**Phase 3 — Smeared spectral functions.** Resolution enters the query type itself, in the Hansen–Lupo–Tantalo mold: Euclidean correlator data in, certified smeared spectral answers out. First contact with real lattice-QCD workflows and real error budgets.

**Phase 4 — Coarse-grained dynamics.** ML-assisted discovery of Mori–Zwanzig closures: automatic identification of slow variables plus certified (or honestly statistical) memory-kernel truncations. Hardest tier — nonlinear, time-dependent error bounds are brutal, and the type system exists precisely so this phase can ship empirical certificates without pretending otherwise.

## Non-goals

Not a monolithic simulation code — the compiler targets existing engines rather than replacing them. Not a surrogate-model factory — uncertified approximations are abundant and are not the gap. Not a complexity-theory jailbreak — BQP-completeness, QMA-hardness, and the sign problem are boundaries of the search space; the compiler's job is to map the compressible islands, not to deny the ocean.

## What success looks like

**Near term:** the compiler rediscovers FMM and a directional Helmholtz variant from black-box kernels, with machine-checked end-to-end bounds. **Mid term:** a certified cluster-expansion or smeared-spectral pipeline is used in a published calculation by a group unaffiliated with this project. **Long term:** a growing rewrite library in which a compression contributed by one field transfers automatically to another — measured, unromantically, in flops saved on real HPC allocations.

## Intellectual lineage

Greengard & Rokhlin (FMM); Ying, Biros & Zorin and Fong & Darve (kernel-independent/black-box FMM); Candès–Demanet–Ying and successors (butterfly factorizations); Halko–Martinsson–Tropp (randomized NLA); Frigo & Johnson (FFTW), Püschel et al. (Spiral), Ragan-Kelley et al. (Halide); Panchekha et al. (Herbie), Darulova & Kuncak (Rosa); Berz & Makino (Taylor models); Sethna and collaborators (sloppy models); Kohn (nearsightedness); White (DMRG); Lüscher (finite-volume method); Hansen, Lupo & Tantalo (smeared spectral functions); Mori & Zwanzig (projection operators); Huang, Kueng & Preskill (classical shadows); Fawzi et al. (AlphaTensor), Romera-Paredes et al. (FunSearch), Novikov et al. (AlphaEvolve).

## Contributing

The project needs people who sit at uncomfortable intersections: numerical analysts who read PL papers, physicists who can state what their community's error bounds actually depend on, proof-assistant engineers interested in analysis rather than algebra, and skeptics who enjoy breaking certificates. The first useful contribution is an attack: a rewrite whose claimed bound fails to compose, filed as an issue with a counterexample.

*The fast multipole method was the existence proof that the map from model to answer is compressible. This project is the induction step.*
