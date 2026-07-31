# Targets

Applied domains where the compiler should earn its keep. Companion to `VISION.md`, whose vocabulary — rewrite, certificate, and the **rigorous / asymptotic / empirical** guarantee tiers — is assumed throughout.

## How a domain earns a slot

1. **The queries are low-dimensional functionals with tolerances the community already states.** Chemical accuracy, noise-weighted mismatch, FLAG-style error budgets — a target is only as good as its habit of declaring ε.
2. **A physical contraction with quantifiable provenance exists.** A gap, a small parameter, a light cone, a convergence radius, a resolution scale — something a certificate can be hung on.
3. **At least one artisanal compression already works.** These become seed rewrites and regression tests; per `VISION.md`, rediscovery is the test suite.
4. **A Python-facing ecosystem exists to compile against.** The compiler targets existing engines; it does not rewrite a field's codes.

Targets are ordered roughly by proximity to the Phase 0–3 machinery.

## Wave scattering and computational electromagnetics

The nearest beachhead, because it is Phase 1 wearing engineering clothes. The queries are S-parameters, radar cross sections, antenna patterns, and photonic band structures — a handful of numbers per design, demanded across thousands of parameter points in inverse-design loops. The contraction is rank structure: hierarchical low-rank blocks for Laplace-like regimes, directional and butterfly structure in the high-frequency Helmholtz regime, and certified reduced bases (Patera–Rozza a posteriori estimators) across parameter sweeps where inf-sup constants cooperate. MLFMA and butterfly-compressed boundary elements are the artisanal compressions to rediscover; the compiler's addition is automatic selection among H-matrix, butterfly, and reduced-basis rewrites per frequency band, with a composed bound feeding the design loop. Certificate outlook: rigorous to asymptotic — the friendliest terrain we have. Ecosystem hooks: FEniCSx, MEEP, scikit-rf.

## Plasma kinetics and fusion energy

Nobody wants f(x, v, t); everyone wants its moments — turbulent heat and particle fluxes, growth rates, transport coefficients, stability margins. The field is organized around a hierarchy of small-parameter reductions (kinetic → gyrokinetic → gyrofluid → MHD) with explicit expansion parameters like the normalized gyroradius ρ* ~ 10⁻³ and collisionality, which is to say: asymptotic certificates already exist, hand-derived. Dynamical low-rank approximation for Vlasov (Einkemmer–Lubich) brings genuine, if partial, error analysis; the Hammett–Perkins Landau-fluid closure is a hand-built Mori–Zwanzig projection and a canonical Phase 4 regression test. The compiler's addition: certified dispatch along the reduction hierarchy per query tolerance, and closure discovery with an honestly declared empirical tier — exactly what stellarator optimization loops need, since they can only afford query-budgeted physics. Certificate outlook: asymptotic for the hierarchy, empirical for discovered closures. Ecosystem hooks: pyrokinetics, PlasmaPy, DESC.

## Quantum chemistry and materials

The flagship for rigorous intervals. Queries come with a declared tolerance culture — chemical accuracy, 1 kcal/mol — and the contraction is everywhere: Kohn nearsightedness in gapped systems, locality of correlation (DLPNO-class methods reach near-linear scaling), and pervasive low-rank structure in the two-electron integrals (density fitting, Cholesky, tensor hypercontraction). The headline product writes itself: a variational wavefunction gives an upper bound, a variational 2-RDM under N-representability conditions gives an SDP lower bound, and the pair yields a *certified bracket on the ground-state energy within a basis* — with basis-set incompleteness then handled at the asymptotic tier (extrapolation) or empirical tier (F12), declared, not hidden. The compiler's addition is replacing the folklore method ladder with per-system, per-query selection of thresholds and rewrites against a composed error budget. Certificate outlook: rigorous intervals where the sandwich closes; declared-empirical DFT elsewhere. Ecosystem hooks: PySCF, ASE.

## Lattice field theory

Home turf for Phases 2–3, extended to the queries the community actually fights over. Window quantities for the muon g−2 hadronic vacuum polarization and HLT-smeared spectral functions are resolution-limited queries by construction; multigrid Dirac solvers, multilevel Monte Carlo via domain factorization (Cè–Giusti–Schaefer), and master-field techniques are the existing hierarchy-exploiting rewrites. The field's error-budget culture (FLAG averages institutionalize it) makes adoption plausible: the compiler's addition is composing statistical sampling error, solver tolerances, and continuum/volume extrapolations into one end-to-end budget per declared query. Certificate outlook: statistical certificates with rigorous components; the sign wall at finite density is a mapped boundary, not a promise. Ecosystem hooks: gpt (Grid Python Toolkit).

## Nuclear structure and dense matter

Chiral EFT hands us the rare thing: an expansion parameter whose truncation error is *modelable* — the BUQEYE program turns EFT convergence patterns into statistical certificates with declared priors. On the many-body side, eigenvector continuation revealed that parametric nuclear solves live on tiny reduced bases, with a posteriori error estimation emerging from its reformulation as a reduced-basis method; SRG evolution is a softening rewrite that trades operator complexity for tractability. The queries with leverage are binding energies, astrophysical reaction rates, and neutron-star equation-of-state bands, where certified emulators must propagate EFT truncation error all the way to observables. Certificate outlook: statistical and asymptotic, unusually well-founded for a strongly correlated field. Ecosystem hooks: the BUQEYE toolkits.

## Molecular kinetics, free energies, and coarse-graining

The queries are thermodynamic and kinetic functionals — binding free energies, solvation, rates, mechanisms — at kcal/mol tolerances. Metastability is a spectral gap of the transfer operator, so Markov state models are projections whose quality is bounded by variational scores (VAC/VAMP); MBAR carries asymptotic variance estimates; polymer and solvent coarse-graining is Mori–Zwanzig in production use. The compiler's addition: automated slow-variable identification with variational certificates on the projection, and — critically — an honest empirical tier for force-field error, which dominates real budgets and is today almost never propagated. This is Phase 4 territory with the best-developed variational scaffolding. Certificate outlook: variational bounds on kinetics, statistical on sampling, declared-empirical on the model itself. Ecosystem hooks: OpenMM, deeptime, pymbar.

## Quantum dynamics and near-term quantum hardware

The query with commercial teeth is dispatch: given a circuit, an observable, a time, and a noise model, does a classical path exist — and if so, with what certificate? The compressible islands are mapped and growing: Lieb–Robinson cones bound operator spreading rigorously; noise truncates Pauli-path expansions (the mechanism behind the classical reproduction of utility-scale experiments and the Aharonov et al. dequantization); stabilizer-rank decompositions (Bravyi–Gosset) cover near-Clifford circuits; process tensors are low-rank in memory time (TEMPO); classical shadows budget the measurement side. The compiler's addition is turning this hardness cartography into a decision procedure: certified classical answer when one exists, quantum resource estimate when not. Certificate outlook: rigorous on the islands, empirical for heuristic tensor networks in between — with the tier printed on every answer. Ecosystem hooks: QuTiP, stim, quimb.

## Turbulence and engineering transport

Eddies are detail; the queries are means — drag coefficients, Nusselt numbers, mixing rates. Two complementary seams. First, the background-flow method in its modern sum-of-squares form delivers *rigorous* bounds on mean quantities via convex optimization (Doering–Constantin lineage through Tobasco, Fantuzzi, Goulart): certificates that are literally the product, on a problem famous for having none. Second, RANS and subgrid closures are Mori–Zwanzig instances currently shipped with silent extrapolation; automated closure discovery with declared validation domains is the honest upgrade. The compiler's addition is pairing the two, so every reported mean carries either a bound or a tier. Certificate outlook: rigorous bounds on one side, declared-empirical closures on the other, and no pretense they meet in the middle yet. Ecosystem hooks: Dedalus.

## Gravitational-wave source modeling

The purest resolution-limited query in physics: waveforms need only match to the detector's noise-weighted mismatch, a tolerance set by instrument and SNR, not by formalism. Reduced-order surrogates of numerical relativity (empirical interpolation over a low-dimensional waveform manifold) and post-Newtonian/EOB expansions (asymptotic tier, explicit small parameters) are the existing compressions; accuracy studies against NR are the current, largely empirical, certification practice. The compiler's addition is mismatch-budgeted surrogate synthesis with interpolation-error certificates where the manifold structure permits them — query-first accounting as the literal design spec. Certificate outlook: empirical with asymptotic components, and an unusually crisp ε. Ecosystem hooks: bilby, gwsurrogate.

## Further afield

The same archetypes recur wherever detail is simulated but functionals are wanted: combustion mechanism reduction (CSP/DRG against ignition-delay queries), climate and weather subgrid closures, neutron transport and reactor design, semiconductor device modeling. Each is admissible the moment it satisfies the four criteria above; none is a launch target.

## Recurring certificate archetypes

Four patterns cover nearly everything above, and the rewrite library should be organized around them: **small-parameter expansions** (gyrokinetics, EFT, post-Newtonian, Braginskii — asymptotic tier); **variational sandwiches** (chemistry's upper/lower bracket, SOS transport bounds, reduced-basis a posteriori estimators — rigorous tier); **resolution-limited queries** (HLT smearing, g−2 windows, GW mismatch — ε supplied by the instrument); and **projection with memory** (Mori–Zwanzig in MD, plasma closures, subgrid models — empirical tier, with variational scores where they exist). A compression contributed under one archetype in one field should transfer to its siblings mechanically; that transfer is the flywheel working.

## Sequencing

| Target | Roadmap phases | Dominant contraction | Certificate outlook |
|---|---|---|---|
| Wave scattering & CEM | 1 | rank structure, butterflies, reduced bases | rigorous / asymptotic |
| Plasmas & fusion | 2–4 | small-parameter hierarchy, closures | asymptotic → empirical |
| Quantum chemistry & materials | 2–3 | nearsightedness, low-rank ERIs, variational sandwich | rigorous intervals + declared-empirical |
| Lattice field theory | 2–3 | hierarchy, factorization, smearing | statistical + rigorous parts |
| Nuclear & dense matter | 2–3 | EFT expansion, reduced bases | statistical / asymptotic |
| Molecular kinetics & free energy | 4 | spectral-gap projection, Mori–Zwanzig | variational + declared-empirical |
| Quantum dynamics & devices | 2–4 | light cones, noise truncation, stabilizer rank | rigorous islands → empirical |
| Turbulence & transport | 4 | SOS bounds, subgrid closures | rigorous bounds + declared-empirical |
| Gravitational waves | 3 | ROM manifolds, PN/EOB expansions | empirical + asymptotic |

Beachhead order: CEM first, as the Phase 1 shakedown against MLFMA and butterfly baselines; the chemistry energy bracket as the first headline rigorous interval; lattice windows and the plasma reduction hierarchy as Phase 2–3 partners, chosen because both communities already budget error; the quantum dispatch procedure and SOS transport bounds as certificate showpieces; molecular and plasma closure discovery last, carrying Phase 4's declared-empirical banner. The selection logic throughout: go where ε is already spoken aloud, so the compiler's certificates land in a culture that knows how to read them.

*Every phase of the roadmap should have a paying customer.*
