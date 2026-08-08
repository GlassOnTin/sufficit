# Targets

Applied domains where the compiler should earn its keep. This is a companion to
`VISION.md`, whose vocabulary is assumed throughout: rewrite, certificate, and
the **rigorous / asymptotic / empirical** guarantee tiers.

## How a domain earns a slot

1. **The queries are low-dimensional functionals with tolerances the community
   already states.** Chemical accuracy, noise-weighted mismatch, FLAG-style
   error budgets. A target is only as good as its habit of declaring ε.
2. **A physical contraction with quantifiable provenance exists.** A gap, a
   small parameter, a light cone, a convergence radius, a resolution scale.
   Something a certificate can be hung on.
3. **At least one artisanal compression already works.** These become seed
   rewrites and regression tests. Per `VISION.md`, rediscovery is the test
   suite.
4. **A Python-facing ecosystem exists to compile against.** The compiler
   targets existing engines. It does not rewrite a field's codes.

Targets are ordered roughly by proximity to the Phase 0 to 3 machinery.

## Wave scattering and computational electromagnetics

The nearest beachhead, because it is Phase 1 wearing engineering clothes. The
queries are S-parameters, radar cross sections, antenna patterns, and photonic
band structures. Each is a handful of numbers per design, demanded across
thousands of parameter points in inverse-design loops.

The contraction is rank structure. There are hierarchical low-rank blocks for
Laplace-like regimes, directional and butterfly structure in the high-frequency
Helmholtz regime, and certified reduced bases across parameter sweeps where
inf-sup constants cooperate (Patera-Rozza a posteriori estimators). MLFMA and
butterfly-compressed boundary elements are the artisanal compressions to
rediscover.

The compiler's addition is automatic selection among H-matrix, butterfly and
reduced-basis rewrites per frequency band, with a composed bound feeding the
design loop. Certificate outlook: rigorous to asymptotic, which is the
friendliest terrain we have. Ecosystem hooks: FEniCSx, MEEP, scikit-rf.

## Plasma kinetics and fusion energy

Nobody wants f(x, v, t). Everyone wants its moments: turbulent heat and
particle fluxes, growth rates, transport coefficients, stability margins.

The field is organized around a hierarchy of small-parameter reductions,
running kinetic to gyrokinetic to gyrofluid to MHD, with explicit expansion
parameters such as the normalized gyroradius ρ* ~ 10⁻³ and collisionality. That
is to say asymptotic certificates already exist, hand-derived. Dynamical
low-rank approximation for Vlasov (Einkemmer-Lubich) brings genuine, if
partial, error analysis. The Hammett-Perkins Landau-fluid closure is a
hand-built Mori-Zwanzig projection and a canonical Phase 4 regression test.

The compiler's addition is certified dispatch along the reduction hierarchy per
query tolerance, plus closure discovery with an honestly declared empirical
tier. That is exactly what stellarator optimization loops need, since they can
only afford query-budgeted physics. Certificate outlook: asymptotic for the
hierarchy, empirical for discovered closures. Ecosystem hooks: pyrokinetics,
PlasmaPy, DESC.

## Quantum chemistry and materials

The flagship for rigorous intervals. Queries come with a declared tolerance
culture, namely chemical accuracy at 1 kcal/mol, and the contraction is
everywhere: Kohn nearsightedness in gapped systems, locality of correlation
(DLPNO-class methods reach near-linear scaling), and pervasive low-rank
structure in the two-electron integrals (density fitting, Cholesky, tensor
hypercontraction).

The headline product writes itself. A variational wavefunction gives an upper
bound. A variational 2-RDM under N-representability conditions gives an SDP
lower bound. The pair yields a *certified bracket on the ground-state energy
within a basis*. Basis-set incompleteness is then handled at the asymptotic
tier by extrapolation, or at the empirical tier by F12, and it is declared
rather than hidden.

The lower half of that pair now exists. The 2-positivity relaxation is
solved and certified in `rdm2_energy_bracket`, and spin blocking carries it to
H₁₄, which is 2²⁸ Fock states and well past the point where the Hamiltonian
can be formed. It certifies the N-electron sector rather than the whole Fock space,
which is a narrower claim than the bracket beside it. Its upper half is a
self-consistent determinant, so the sandwich is assembled from two proofs that
know nothing about each other, and the planner races it against the window
ladder. There is now a molecular front door too, on McMurchie-Davidson
integrals and published STO-3G for H, C, N and O. Polarisation functions push
a six-atom hydrogen cluster past the point where its Fock sector can be formed
and only the relaxation answers, and on water, where the exact answer is still
affordable, the relaxation lands 2.3 mHa below full CI.

The compiler's addition is replacing the folklore method ladder with
per-system, per-query selection of thresholds and rewrites against a composed
error budget. Certificate outlook: rigorous intervals where the sandwich
closes, and declared-empirical DFT elsewhere. Ecosystem hooks: PySCF, ASE.

## Lattice field theory

Home turf for Phases 2 and 3, extended to the queries the community actually
fights over. Window quantities for the muon g−2 hadronic vacuum polarization,
and HLT-smeared spectral functions, are resolution-limited queries by
construction. Multigrid Dirac solvers, multilevel Monte Carlo via domain
factorization (Cè-Giusti-Schaefer), and master-field techniques are the
existing hierarchy-exploiting rewrites.

The field's error-budget culture makes adoption plausible, since FLAG averages
institutionalize it. The compiler's addition is composing statistical sampling
error, solver tolerances, and continuum and volume extrapolations into one
end-to-end budget per declared query. Certificate outlook: statistical
certificates with rigorous components. The sign wall at finite density is a
mapped boundary, not a promise. Ecosystem hooks: gpt (Grid Python Toolkit).

## Nuclear structure and dense matter

Chiral EFT hands us the rare thing, an expansion parameter whose truncation
error is *modelable*. The BUQEYE program turns EFT convergence patterns into
statistical certificates with declared priors.

On the many-body side, eigenvector continuation revealed that parametric
nuclear solves live on tiny reduced bases, with a posteriori error estimation
emerging from its reformulation as a reduced-basis method. SRG evolution is a
softening rewrite that trades operator complexity for tractability.

The queries with leverage are binding energies, astrophysical reaction rates,
and neutron-star equation-of-state bands, where certified emulators must
propagate EFT truncation error all the way to observables. Certificate outlook:
statistical and asymptotic, unusually well-founded for a strongly correlated
field. Ecosystem hooks: the BUQEYE toolkits.

## Molecular kinetics, free energies, and coarse-graining

The queries are thermodynamic and kinetic functionals at kcal/mol tolerances:
binding free energies, solvation, rates, mechanisms.

Metastability is a spectral gap of the transfer operator, so Markov state
models are projections whose quality is bounded by variational scores (VAC and
VAMP). MBAR carries asymptotic variance estimates. Polymer and solvent
coarse-graining is Mori-Zwanzig in production use.

The compiler's addition is automated slow-variable identification with
variational certificates on the projection. It also adds, critically, an honest
empirical tier for force-field error, which dominates real budgets and is today
almost never propagated. This is Phase 4 territory with the best-developed
variational scaffolding. Certificate outlook: variational bounds on kinetics,
statistical on sampling, declared-empirical on the model itself. Ecosystem
hooks: OpenMM, deeptime, pymbar.

### The standing challenge: water's density maximum

The named goal for this column is to certify the temperature of maximum
density of a declared water model. Water is densest at about 3.98 °C at one
atmosphere, because the collapse of the open tetrahedral hydrogen-bonded
network competes with ordinary thermal expansion, and below that temperature
the collapse wins. It is the sharpest test of a force field anyone routinely
runs, and it is a good target here for three separate reasons.

It punishes the model specifically. Rigid three-site models miss the maximum by
tens of kelvin, and TIP3P has no maximum in the liquid range at all.
TIP4P/2005 and TIP5P land within a degree or two, on the same functional form
and at the same cost. Getting the density right at 25 °C tells you almost
nothing about whether a model will do this.

It is a sampling problem before it is a physics problem. Between 0 and 4 °C the
density moves by about one part in ten thousand, so the statistical error on an
NPT average has to sit well below that before the question can even be asked.

And the query is a stationary point, not a value, which changes what has to be
certified. Locating a maximum means certifying an ORDERING: rho(T1) < rho(T2) >
rho(T3), with the certified intervals disjoint enough that the ordering is
forced. The precision needed is set by the curvature of rho(T) near the peak
rather than by any tolerance stated up front, and precision at the peak trades
against width in T. That is a well-posed query in this library's terms and a
better-behaved one than asking for the density itself.

The error budget splits three ways and only two of them are ours. Statistical
error from finite sampling is certifiable, at the empirical tier, with a
printed failure probability. Finite-size error is a system-size ladder read by
the same grid-convergence machinery the reactor, the junction and the tokamak
already share, so it is an archetype transfer rather than new work. Force-field
error is neither, and it dominates both by orders of magnitude. So the
achievable claim is "the maximum-density temperature of TIP4P/2005 is 278 +- 2
K", and the unachievable one is "the maximum-density temperature of water is
277.13 K". The gap between those two sentences is exactly what the tier system
exists to declare rather than hide, and here it is tens of kelvin wide
depending on which model was declared.

The blocks, cheapest and most reusable first. The first two need no molecular
dynamics and no new dependency, and both can be gated against a truth known by
construction, so the certification machinery gets tested before an engine is
involved.

1. **A certified mean of a correlated time series. DONE**, as
   `timeseries_mean`. The bottleneck, and needed by every molecular-dynamics
   observable rather than just this one. An NPT average is over autocorrelated
   samples, so bounding it needs an effective sample size and a statement that
   survives correlation. Rigorously that wants a spectral gap nobody will have,
   so it lands at the empirical tier with a declared stationarity assumption.
   Measured over 300 AR(1) replicas of 20,000 samples, coverage runs 0.946
   against a nominal 0.95 at correlation times of 1, 9 and 39, and about 1% of
   series are refused.

   The refusal is the interesting half. The integrated autocorrelation time is
   estimated on a ladder of batch sizes, and if it has not plateaued the series
   is too short to bound the mean at all, which is the same refusal the sea
   wall makes when a resolution ladder shows no asymptotic range. Two
   measurements changed how that test is written. A fixed plateau tolerance
   fired on estimator noise and refused a quarter of INDEPENDENT series, so the
   threshold now carries the noise of the rungs being compared. And the plateau
   test alone was still too weak: at a correlation time of 199, where the
   longest affordable batch is about three correlation times, it let 64% of
   series through and those covered 0.901 rather than 0.95. It is only a proxy
   for what actually matters, so the direct requirement is now stated as well,
   that a batch be at least ten correlation times long. That refuses the whole
   under-covering regime, 300 out of 300. A drifting series, which is what an
   unequilibrated trajectory looks like, is refused for the same reason.
2. **A certified stationary point. DONE**, as `argmax_bracket`. The ordering
   argument above. Under a declared unimodality a pair of disjoint intervals
   with the lower one on the left puts the peak to the right of it, and the
   mirror argument bounds it from above, so the bracket runs from the largest
   abscissa certifiably left of the peak to the smallest certifiably right of
   it. Overlapping error bars establish nothing and buy no bracket. Failure
   probabilities add by union bound and the tier is the weakest consulted. A
   lower bound at or above the upper one is raised rather than reported, since
   it is evidence that the unimodality declaration is false.

   Gated on a synthetic curve with the maximum put in by hand, sampled through
   correlated series so blocks 1 and 2 compose the way they will have to for a
   real trajectory. It brackets the known peak 60 times out of 60 at a noise
   level where the orderings separate, and refuses 52 times out of 60 where
   they do not.

   The cost of the query is now measured rather than guessed, and it is
   steep. Near a stationary point the function is flat, so the orderings are
   hardest exactly where they are wanted, and resolving the peak to within
   delta needs a precision going as delta squared. On a 0.25 K grid, 20,000
   samples per temperature give a density half-width of 7.2e-5 and a 3.94 K
   bracket, while 1,280,000 give 8.6e-6 and 1.90 K. Sixty-four times the
   sampling bought a bracket two times narrower. Anyone planning this should
   see that number before buying the compute rather than after.

   One bug is worth recording because the gate is what caught it. The upper
   comparison was inverted, which put the bracket on the wrong side while
   still returning something plausible, and it showed up as coverage of zero
   and, oddly, tighter noise refusing MORE often. Reading the docstring would
   never have found it, because the docstring described the correct rule.
3. **A finite-size ladder. DONE**, and it turned out to be a transfer rather
   than a build. A ladder of certified values at decreasing 1/N is the same
   object as a ladder at decreasing mesh spacing, so `continuum_limit` reads
   it with no change at all: the function that lifts a reactor off its mesh
   and a junction off its grid lifts a molecular simulation off its box size.
   That is the flywheel claim doing work rather than being asserted.

   The finite-size parameter is 1/N rather than 1/L on purpose. The leading
   correction goes as 1/N, so in that variable the formal order is 1 and sits
   under the cap `gci_extrapolate` applies. In 1/L the same correction is
   third order and would be capped to 2, which only loosens the bound but
   reports an order the physics does not have. The ladder has to be geometric
   because Roache's method needs a fixed refinement ratio, and each rung is
   eight times the molecules, which is what doubling a box costs.

   Gated on a synthetic ladder with the infinite-size limit known by
   construction. It reads the ladder and contains that limit, and the refusal
   boundary is exactly the factor-of-ten precondition: the closest pair of
   rungs differ by 1.1e-4, so it wants a statistical half-width below 1.1e-5,
   and at a per-frame noise of 1e-4 the half-width is 6.9e-6 and it reads
   while at 3e-4 it is 2.1e-5 and it refuses.

   That boundary is where the real difficulty of this whole challenge sits.
   In a molecular simulation the sampling error and the finite-size shift are
   routinely the same size, and when they are, no extrapolation is honest.
4. **An untrusted molecular-dynamics engine.** The certifying half is done, as
   `ensemble_check`, and it turned out to be the substance of this block. An
   engine is untrusted the way every other engine here is untrusted, and that
   means nothing checks the integrator, because nothing can from the outside.
   What can be checked is whether the trajectory is consistent with the
   ensemble the engine claimed, and there are only two questions.

   Is each series stationary at all? `timeseries_mean` already answers that: an
   unequilibrated or leaking run has an autocorrelation time that never
   plateaus, so the mean cannot be bounded and it refuses. An integrator
   leaking energy is caught there, and no separate drift test is needed. The
   diagnosis matters as much as the catch, because the run is refused as
   unusable rather than as biased, which points at the integrator rather than
   at the setpoint.

   And did the controlled variables come back where they were put? A thermostat
   set to 298 K whose certified mean temperature excludes 298 K is not sampling
   that ensemble, whatever it reports, and everything measured downstream is
   about some other system.

   The threshold for throwing a run away is deliberately much stricter than the
   one for reporting an interval, because rejecting is the expensive action and
   every controlled variable is another chance to trip it. Measured over 300
   good two-setpoint runs: 7.7% thrown away at 0.05 against 1.7% at 1e-3, and
   all of that 1.7% is `timeseries_mean` declining to bound a series rather than
   the setpoint test, which did not falsely reject once in 300. A thermostat 2%
   hot is caught 60 times out of 60.

   What remains is attaching a real engine, which is a dependency decision
   rather than a code one. OpenMM is the obvious choice and is packaged both by
   apt and by pip. It should follow the pattern this document already sets for
   Basilisk: drive it as an untrusted external engine, and expect recorded-run
   pages rather than CI regeneration at first, since the CI job is already
   carrying FEniCSx.

   What none of this checks is the force field, which is the error that
   dominates everything in this challenge and is declared rather than
   certified. A sampler can pass every test above while describing a liquid
   that is not water.
5. **The query end to end. DONE**, as `water_tmd_bracket`, against a model
   whose maximum-density temperature is published. The engine is mW, Molinero
   and Moore's monatomic water: a
   Stillinger-Weber solid potential reparameterised so that one site per
   molecule, with no charges and no hydrogen, builds water's tetrahedral
   network out of the three-body term alone. No electrostatics means no Ewald
   sum and no hydrogens means no rigid-body constraints, so it is a page of
   numpy rather than a dependency, which is why it is here and not OpenMM.

   It was chosen because it is published, not because it is cheap. mW has a
   density maximum at 250 K in the literature, so the bracket either contains
   that number or the code is wrong. A model invented here would have proved
   nothing.

   The front door, `water_tmd_bracket`, contains no new machinery, and that is
   the point of it. `ensemble_check` asks whether each trajectory is usable and
   landed on its thermostat, `timeseries_mean` bounds each density paying for
   correlation, and `argmax_bracket` turns the certified orderings into a
   bracket. Blocks 1, 2 and 4 compose and nothing else is needed.

   Four gates on the engine before it was allowed to answer anything, two of
   them independent of any run:

   - Analytic forces against a central difference of the energy, worst
     component 7e-9 kcal/mol/A. The three-body force has four terms that are
     easy to get subtly wrong and a wrong one still integrates plausibly.
   - On a perfect diamond lattice every angle is exactly tetrahedral, so the
     three-body term must vanish identically. It does, to 1e-13, and the total
     energy equals the pair sum to ten digits with no net force. That is a
     deterministic gate on the fiddliest code in the engine: the padded
     neighbour lists, the mask that keeps padding out of an exponential which
     diverges at the cutoff, the triplet enumeration and the sign convention on
     the angle. The lattice does the rest, since the first shell sits at 2.6884
     A with exactly four neighbours and the second at 4.3901 A, just outside
     the 4.3065 A cutoff.
   - Energy conservation without a thermostat: 8e-4 kcal/mol per molecule of
     drift over 10 ps at a 5 fs step.
   - The cohesive energy of the liquid, which is fixed independently. mW is
     fitted to an enthalpy of vaporisation of 10.65 kcal/mol, putting the
     liquid at RT - 10.65 = -10.06 kcal/mol per molecule. The run gives -10.071
     at 64 molecules and -10.080 at 216.

   The density at 298 K is the fourth gate and it needs a caveat the others do
   not. At 216 molecules the run gives 0.9987 g/cm^3 against a published 0.997,
   which is a match. At 64 it gives 0.9908, which is not. The cohesive energy
   is insensitive to size and the density is not, and that 0.8% shift is about
   the size of the whole density change across the temperature range this
   challenge is about. So the cheap box is cheap for a reason.

   Three things had to be measured rather than assumed, and all three were
   quiet failures rather than loud ones.

   The timestep is 5 fs, not the 10 fs mW is usually run at. At 10 fs the
   kinetic temperature comes out 294.73 K when 298 was asked for, and 294.62 at
   a friction five times larger, so it is discretisation and not thermostat
   coupling. At 5 fs it is 298.06 and 298.33. That 1.1% miss is what
   `ensemble_check` refuses, and it matters that the density was the same at
   both timesteps: BAOAB's configurational sampling is accurate enough at 10 fs
   that the answer would have looked fine. The certificate chose the timestep,
   not the folklore.

   The barostat is Monte Carlo on the logarithm of the volume, which needs no
   virial and therefore no pressure estimator, so the pressure is imposed
   exactly rather than measured and controlled. Its move size has to be scaled
   as 1/sqrt(N) to track the natural volume fluctuation. Getting it wrong is
   silent: at 95% acceptance the lag-one correlation of the density series is
   0.994, the volume barely moves, and the run looks converged while sampling
   almost nothing. At 53% acceptance it is 0.648.

   And the run has to be melted first. The starting diamond lattice IS mW's ice
   structure, and a defect-free crystal under periodic boundaries has no
   interface to melt from, so it superheats indefinitely. Started cold at 298
   K, 23 K above the model's 274.6 K melting point, it stayed crystalline:
   density 0.9764 at 64 molecules and 0.9768 at 216, so not a finite-size
   artefact, at a potential energy of -11.39 kcal/mol per molecule. Melting
   first gives 0.9908 and -10.071. The 1.3 kcal/mol difference is the heat of
   fusion, and without the independent enthalpy-of-vaporisation gate above
   there would have been no way to tell which number was the liquid.

   That last one is the sharpest result of this block, because it marks the
   boundary of what block 4 can do. Run long enough to bound its own series,
   15,000 samples, the crystal PASSES `ensemble_check`: density 0.97640 +-
   0.00055 with its halves at 0.9765 and 0.9763, an autocorrelation time of
   4.31 samples plateauing at 1.09, and a kinetic temperature of 295.79 +- 2.35
   K containing the 298 K it was set to. Every question block 4 knows how to
   ask gets the right answer, on a perfectly stationary and perfectly
   well-sampled trajectory of the wrong phase.

   Consistency with the declared ensemble is not ergodicity. Nothing visible in
   a trajectory reveals a basin it never left, and the check that caught this
   had to come from outside: a number the model was fitted to. That is a
   general limit and not an mW one. It is the same shape as the force-field
   caveat, which no amount of sampling reaches either.

   One design decision came out of the first production rung rather than out of
   argument. At 210 K the density series is still drifting after 3 ns, halves
   at 0.98789 and 0.98293, and `timeseries_mean` refuses it: the
   autocorrelation time is still growing by 2.70x against a 1.80 tolerance.
   That is correct, since deeply supercooled mW is viscous and slow to forget
   the configuration it was quenched from, and it is also the coldest rung,
   which is the one that would have bounded the peak from below.

   As first written, that one refusal killed the whole query. It should not. A
   series that cannot be bounded is a statement about one run and can be
   answered by dropping that temperature; a missed setpoint is a statement
   about the engine and dropping it would hide the problem. So the two are now
   separate exceptions, an unbounded rung is dropped and named in the
   provenance, a missed setpoint is still fatal, and below three surviving
   rungs it refuses and lists what was lost.

   Quenching each temperature straight off the melt was also wrong, and the
   ladder is now annealed downward from a single melt so each rung starts from
   an equilibrated liquid 20 K warmer. That fixed the warm half outright. At 64
   molecules and 2 ns per rung:

       290 K   rho = 0.99941 +- 0.00148
       270 K   rho = 1.00035 +- 0.00185
       250 K   rho = 1.00296 +- 0.00189
       230 K   refused, halves 1.00086 / 0.99526
       210 K   refused, halves 0.99017 / 0.97233
       370 K   rho = 0.97933 +- 0.00179

   The 210 K refusal is not a numerical failure, it is the answer. The density
   falls through the run from 0.990 towards 0.972, which is this model's ice
   density, so the trajectory is crystallising and there is no stationary
   liquid mean to report. That is where the sampling problem this challenge is
   about turns into a phase-transition problem, and the refusal says so instead
   of averaging across it.

   At 230 K the drift was equilibration rather than freezing, and it went away
   when the run was made long enough: refused at 2 ns, bounded at 7.5 ns with
   its halves agreeing at 0.99900 and 0.99891. The lower bound on the peak is
   the expensive side of this query, and it is expensive for a physical reason
   rather than a statistical one. Spending the budget where it was needed
   rather than evenly across the ladder is what closed it.

   The certified ladder, mW at 64 molecules and 1 atm, over three separately
   annealed chains, each state point equilibrated and checked on its own:

       230 K   0.99896 +- 0.00197        290 K   0.99941 +- 0.00148
       240 K   1.00042 +- 0.00168        310 K   0.99561 +- 0.00156
       250 K   1.00296 +- 0.00189        330 K   0.98966 +- 0.00195
       270 K   1.00035 +- 0.00185        350 K   0.98383 +- 0.00154
                                         370 K   0.97933 +- 0.00179

       argmax_bracket: [230, 290] K = 260 +- 30 K, EMPIRICAL, fail_p 0.45

   The published mW maximum, 250 K, is inside it. The point estimates peak at
   250 K exactly, which is pleasing and is not the claim; the bracket is.

   Now the part that matters more than the bracket. That fail_p of 0.45 is a
   union bound over nine rungs at 0.05 each, and it is not a strong
   certificate. Asked for an honest overall 5%, meaning 0.0056 per rung, the
   SAME data refuses: the ordering that carries the whole lower bound is the
   one between 230 and 250 K, and it clears by 1.42e-4 in density at 0.05 per
   rung and fails by 1.64e-3 at 0.0056. One part in seven thousand is the
   entire margin. Getting the Bonferroni-honest version needs the two rungs
   either side of that gap tightened by about 1.4x, which is roughly twice the
   sampling, and only at those two.

   So the query closes, at a confidence level worth naming out loud rather than
   burying. Reporting 260 +- 30 K without reporting that it evaporates under
   the correction would be reporting the half that worked.

   The next rung up was out of reach when this was written and is not any
   more, which is worth recording because the reason was algorithmic rather
   than hardware. mW's cutoff holds about eleven neighbours whatever the box,
   so the all-pairs distance matrix the engine started with was almost
   entirely a list of pairs that are not neighbours, and it was 85% of the
   work at 512 sites. Dividing the box into cells a cutoff wide gives, at this
   density, a candidate list of about 189 whatever N is, so the speedup is
   about N/189 and the engine goes from quadratic to linear:

       N=  64   dense   1.02x        N=1000   cell   6.03x
       N= 216   dense   1.00x        N=1728   cell   8.01x
       N= 512   cell    2.65x

   The route is chosen by that arithmetic and not by a rule about box sizes,
   and the factor of two in the guard was earned rather than picked. At 216
   sites the candidate list is already smaller than N, 189 against 216, and
   the cell list was still 12% SLOWER, because a 13% cut in pairs does not pay
   for the sort, the bucket table and the gather. Written the obvious way, the
   optimisation was a 28% regression at exactly the box size the production
   ladders had been using. It now declines itself there.

   That moves a 512-molecule ladder from about 20 hours to about 7, which is
   the difference between a target and an overnight run.

   The 64-molecule box is the other unquantified term, and it has now been
   measured rather than left as a caveat. A 216-molecule ladder, same protocol:

       250 K   1.00480 +- 0.00089
       270 K   1.00253 +- 0.00085
       290 K   1.00003 +- 0.00087
       230 K   refused, halves 0.99926 / 0.99890
       210 K   refused, halves 0.98695 / 0.99055

   Three things came out of it. The half-widths tighten by 1.7x to 2.2x, close
   to the sqrt(216/64) = 1.84 that sigma going as 1/sqrt(N) predicts, at about
   7.6x the cost per step. The upper bound on the peak tightens from 290 K to
   270 K, because the 250 against 270 ordering fails by 1.14e-3 at 64 molecules
   and clears by 5.3e-4 at 216. And the finite-size shift is not a constant
   offset: +0.00184 at 250 K, +0.00218 at 270 K, +0.00062 at 290 K, so the
   density anomaly rho(250) - rho(290) grows from 0.00354 to 0.00477, a third
   stronger in the larger box. Sixty-four molecules under-represent the
   tetrahedral network the anomaly is made of, which means the box size can
   move the peak and not merely the density. The bracket above is certified
   about a box.

   The cold rungs failed differently at the two sizes, and the difference is
   the diagnosis. At 64 molecules 230 K drifted, so it was still equilibrating.
   At 216 it is stationary, halves agreeing to 3.6e-4, and refused purely for
   run length: the plateau test wants 32 batches of at least ten correlation
   times and 2 ns gives 625-sample batches against a correlation time near 62.
   That is a budget, not a physics wall, so the budget was spent: 8 ns at 230 K
   instead of 2. It bounds, at 1.00105 +- 0.00075, with halves at 1.00161 and
   1.00049. It is also 1.9e-3 above what the refused 2 ns run was averaging,
   which is worth noting on its own. The refusal was not conservatism. That run
   really was still equilibrating, and had the plateau test been slightly more
   permissive the ladder would have been handed a number nearly two
   half-widths wrong, in the rung that carries the whole lower bound.

   With it, the 216-molecule query closes, and closes better than the
   64-molecule one did:

       230 K   1.00105 +- 0.00075        270 K   1.00253 +- 0.00085
       250 K   1.00480 +- 0.00089        290 K   1.00003 +- 0.00087

       argmax_bracket: [230, 270] K = 250 +- 20 K

   The published mW maximum is not merely inside this one, it is the centre of
   it. And unlike the 64-molecule bracket, this one survives the honest
   accounting: at 0.05/4 per rung, giving a family-wise 0.05 rather than a
   union bound of 0.20, the bracket is unchanged at [230, 270] K. That is the
   first result in this challenge with a confidence level worth quoting without
   an apology attached.

   The margins say where it is solid and where it is not. The lower bound is
   comfortable, the 230-against-250 ordering clearing by 1.62e-3 in density
   after the correction. The upper bound is on a knife edge: 250 against 270
   clears by 1e-5, which is one part in a hundred thousand. Anything that moved
   it would push the upper bound back to 290 K and the answer to 260 +- 30 K,
   which still contains 250. So the honest reading is a solid lower bound, a
   solid but wider upper bound at 290 K, and a tighter upper bound at 270 K
   that is real but should not be leaned on.

   What made the difference was the box rather than the sampling. The larger
   box tightened the intervals by about a factor of two AND strengthened the
   anomaly by a third, and the ordering that failed at 64 molecules needed
   both.

The rigid-model temperatures quoted at the top of this section are recollection
and are not gated. mW's 250 K is not: it was checked against the literature
before being used as a target, precisely because the whole point of the block
was to have an answer fixed by someone else. This project has been bitten both
ways in one sitting: the STO-3G exponents were recalled correctly and passed
their gate, and the N2 energy was recalled correctly while the code around it
was wrong. Nothing here should be repeated as measured until it has been run.

What the whole ladder does NOT touch, and no part of it ever will, is whether
mW is water. It is a single site with no charge and no hydrogen, fitted to a
melting point, a density and an enthalpy of vaporisation, and it reproduces the
density maximum because it was built to. Every certificate above is about the
model, and the model is declared. That is the same wall the force-field caveat
in block 4 runs into, and it is the one that actually dominates this problem.

## Quantum dynamics and near-term quantum hardware

The query with commercial teeth is dispatch. Given a circuit, an observable, a
time, and a noise model, does a classical path exist, and if so with what
certificate?

The compressible islands are mapped and growing. Lieb-Robinson cones bound
operator spreading rigorously. Noise truncates Pauli-path expansions, which is
the mechanism behind the classical reproduction of utility-scale experiments
and the Aharonov et al. dequantization. Stabilizer-rank decompositions
(Bravyi-Gosset) cover near-Clifford circuits. Process tensors are low-rank in
memory time (TEMPO). Classical shadows budget the measurement side.

The compiler's addition is turning this hardness cartography into a decision
procedure: a certified classical answer when one exists, and a quantum resource
estimate when not. Certificate outlook: rigorous on the islands, empirical for
heuristic tensor networks in between, with the tier printed on every answer.
Ecosystem hooks: QuTiP, stim, quimb.

## Turbulence and engineering transport

Eddies are detail. The queries are means: drag coefficients, Nusselt numbers,
mixing rates. There are two complementary seams.

First, the background-flow method in its modern sum-of-squares form delivers
*rigorous* bounds on mean quantities via convex optimization, following the
Doering-Constantin lineage through Tobasco, Fantuzzi and Goulart. Those
certificates are literally the product, on a problem famous for having none.

Second, RANS and subgrid closures are Mori-Zwanzig instances currently shipped
with silent extrapolation. Automated closure discovery with declared validation
domains is the honest upgrade.

The compiler's addition is pairing the two, so every reported mean carries
either a bound or a tier. Certificate outlook: rigorous bounds on one side,
declared-empirical closures on the other, and no pretense that they meet in the
middle yet. Ecosystem hooks: Dedalus.

## Gravitational-wave source modeling

The purest resolution-limited query in physics. Waveforms need only match to
the detector's noise-weighted mismatch, a tolerance set by instrument and SNR
rather than by formalism.

The existing compressions are reduced-order surrogates of numerical relativity,
built by empirical interpolation over a low-dimensional waveform manifold, and
post-Newtonian and EOB expansions, which sit at the asymptotic tier with
explicit small parameters. Accuracy studies against NR are the current, largely
empirical, certification practice.

The compiler's addition is mismatch-budgeted surrogate synthesis with
interpolation-error certificates where the manifold structure permits them.
That is query-first accounting as the literal design spec. Certificate outlook:
empirical with asymptotic components, and an unusually crisp ε. Ecosystem
hooks: bilby, gwsurrogate.

## Further afield

The same archetypes recur wherever detail is simulated but functionals are
wanted: combustion mechanism reduction (CSP and DRG against ignition-delay
queries), climate and weather subgrid closures, neutron transport and reactor
design, semiconductor device modeling. Each is admissible the moment it
satisfies the four criteria above. None is a launch target.

**Neutron transport** was entered first, ahead of the others and out of order,
because it pays a debt the nine could not. Every eigenvalue bracket in the
library ran on the variational theorem, which needs a self-adjoint operator. A
reactor down-scatters and is not one. The criticality eigenvalue is bracketed
by positivity instead, as the archetype below describes. The field states its
tolerance in pcm, so the certificate lands in units the culture already reads.

**Semiconductor devices** came next, and paid a different debt. Every
certificate here answered *how far is this number from the right one*, and none
could answer *is there a right one*. A device simulator runs Newton's method on
equations whose carrier densities are exponentials of the unknown, and stops
when the residual looks small. That is consistent with a solution nearby, with
a solution far away, and with none at all. Kantorovich's theorem converts a
residual into existence plus an enclosure. The constant it needs most is a
bound on the inverse Jacobian, which the reactor's M-matrix witness already
supplies, because the discretized nonlinear Poisson operator is a Z-matrix too.
Measured on a pn junction, the residual at the last refusing rung is 10⁻³. One
Newton step later it is 5·10⁻⁶ and the certificate closes, at half a per cent
of the answer, because ‖J⁻¹‖ is 237 and the residual shows none of that.

**Combustion mechanism reduction** was probed and **not** entered, which is
worth recording with its numbers rather than leaving as a gap. The headline
query is ignition delay, and a certificate on it must survive the
chain-branching phase. Measured on an eight-species hydrogen mechanism at
1100 K, the spectral abscissa integrates to ∫α dt ≈ 17 over the induction
period. Perturbations are genuinely amplified 2.5·10⁷ times, and that is what
ignition *is*, not a defect of any bound. A fixed-norm Gronwall bound pays far
more than the physical rate, giving ∫μ dt ≈ 3000 in the natural scaling and
≈ 67 after optimal diagonal balancing, because chemical Jacobians are extremely
non-normal. A componentwise comparison enclosure, which pays no norm tax at
all, holds to about 1% through half the induction period and then goes vacuous
at the branching knee. The conclusion is that the field's central query is
exponentially ill-conditioned, so a rigorous forward certificate on ignition
delay is expensive for reasons that are physical. The promising entry is a
quantity that is not exponentially amplified: the crossover limit separating
ignition from no ignition, which is an eigenvalue sign condition on a Metzler
matrix and therefore a cone-preserving bracket.

**Shear-flow stability** was probed next and looks like the strongest unclaimed
entry, for a reason that has little to do with the field and everything to do
with the archetype it needs. The prompt was a photograph: the Inouye Solar
Telescope resolving Kelvin-Helmholtz billows on the Sun at city scale, an
instability described since the 1870s and only recently caught there directly.

The query is the one that field actually asks. Given a velocity profile, is
this layer unstable, and if so how fast does it grow? That is the complex phase
speed c of the Rayleigh equation,

    (U - c)(phi'' - k^2 phi) - U'' phi = 0,

whose operator is neither self-adjoint nor cone-preserving, so archetypes two
and five both miss it. What reaches it is a family of classical integral
identities that confine the answer without discretising anything:

  - **Rayleigh's inflection-point criterion.** No inflection point in U, no
    instability. One pass over U''. A rigorous no.
  - **Fjortoft's criterion.** Strictly stronger, still one pass.
  - **Miles-Howard.** In a stratified layer, a Richardson number at or above
    1/4 everywhere implies stability. A rigorous no, on a quantity the solar
    and oceanographic communities already state tolerances in.
  - **Howard's semicircle theorem.** Every unstable c lies in the upper half of
    the circle centred on the median velocity with radius half the velocity
    range. A rigorous enclosure of a non-self-adjoint eigenvalue, free.

The probe asked the only question that decides whether that is worth having:
how loose is the free bound against what a solver reports. Gating the solver
came first, since a probe that cannot be trusted decides nothing, and it needed
two analytic answers rather than remembered ones. The tanh layer's neutral mode
falls out of substituting phi = sech(z) and c = 0, which leaves
sech^2(z)(1 - k^2) and so is neutral at k = 1 and nowhere else on that branch.
The Bickley jet sech^2(z) has a second, from phi = sech^2(z) with c = 2/3,
where the two coefficients left over vanish together only at k = 2. Against
those, and against Michalke's 1964 numerical integration of the same equation:

    tanh, k=0.445      solver 0.189696, converging as h^2 through
                       0.189610 and 0.189679. Michalke: 0.095|dU|/l_u = 0.19
    Bickley, k->2      solver c_r walks 0.578, 0.632, 0.660 and goes
                       neutral at k=2. Analytic c_r = 2/3
    tanh, k->1         solver loses the mode between k=0.95 and k=0.99,
                       just below the analytic boundary at k=1

That third line is the useful failure. Near the neutral curve the growth rate
falls below what the discretisation resolves, so the eigensolve is weakest
exactly where the question is sharpest, which is the case for having a
criterion that does not discretise at all.

With the solver gated, the bound measured against it:

    tanh, at its most unstable k=0.445          2.3x loose
    Bickley jet, near its peak                  2.5x loose
    Gaussian jet, near its peak                 2.2x loose
    tanh at k=0.8, approaching neutral          7.4x loose
    Bickley at k=1.95, approaching neutral       85x loose

So the free rigorous bound is a factor of about two where the physics lives and
worthless near the neutral curve, and every unstable mode found sat inside the
semicircle. That is the right shape for a dispatch rather than a
disappointment: the identity answers cheaply and coarsely, the eigensolve
answers expensively and precisely, and the certificate records which. It is
also a one-sided bracket and not a sandwich, since nothing here gives a
rigorous lower bound on a growth rate, and that should be said rather than
papered over.

Two limits are real and are not the probe's fault. The semicircle theorem is
proven for inviscid flow and does not survive viscosity, so the certified
object is the Rayleigh problem and the reduction to it is a declared modelling
step. And none of this is yet the MHD problem the solar image poses, where a
field-aligned magnetic field stabilises the layer above a critical Alfven
speed. That extension is the point of entering and it is not measured here.

Criterion four is unusually well satisfied. Dedalus and eigentools already do
spectral linear-stability eigenvalue solves with a resolution-convergence
criterion for rejecting spurious modes, which is this project's proposer and
checker split, already built and already Python.

It is now built, as `shear_stability_bracket`, and it is where the seventh
archetype earns its place: no cone exists here, no symmetry exists, and nothing
else in the library reaches the problem at all. Three certificates and one
solver, in that order of authority.

Rayleigh and Fjortoft come first, are rigorous, are free, and are pointwise on
U. Either firing ends the query at a RIGOROUS zero with no eigenvalue computed:
a parabolic profile and plane Couette by no inflection point, and U = z^3 by
Fjortoft despite having one. They are deliberately evaluated BEFORE the
solver's own preconditions, since a free rigorous answer must not be refused
for a precondition of a solve it never needs.

Howard's semicircle is used as a CHECKER rather than as an answer, which is the
shape this library already has everywhere else. The eigensolve proposes a mode;
if it lands outside the circle of centre (Umax+Umin)/2 and radius
(Umax-Umin)/2, no unstable mode of an inviscid parallel flow can be there, so
the solver is wrong and the query raises instead of reporting. What the solver
contributes is the value at the EMPIRICAL tier, with its error from a second
solve on a coarser mesh, labelled a resolution estimate and not a bound.

Gated against two analytic answers derived rather than recalled, plus one from
the literature: the tanh layer neutral at exactly k=1, from substituting
phi = sech(z) and c = 0; the Bickley jet neutral at k=2 with c=2/3; and
Michalke's 1964 maximum, which the front door reproduces as 0.189506 +-
0.000588 at k=0.445 against 0.19. The Howard cap runs 1.43x loose at k=0.2,
2.35x at the most unstable wavenumber, and 7.47x approaching neutral.

Four bugs are recorded with it because all four were caught by gates rather
than by reading, and two are the same mistake in different clothes. The Bickley
jet's U'' had a sign error and reported a famously unstable jet as stable. The
tanh growth rate was first checked against a remembered k(1-k), which is not
the answer to anything here; the solver was right and the reference was wrong,
which is the N2 episode inverted. And both stability criteria needed a
tolerance relative to the scale the profile COULD have rather than the one it
does: plane Couette has U'' identically zero, so a threshold keyed on max|U''|
collapses and rounding noise reads as sign changes, and on U = z^3 the Fjortoft
product's minimum is -4.5e-21 against a scale of 476, which a strict < 0 test
reads as instability. In both cases a certifiably stable flow was being handed
to the eigensolve. An exact zero is not a small negative number. An ungated
number is not a gate.

**Well-balanced surface tension** is a note rather than a probe, and is marked
as such: nothing below has been measured here, unlike the two entries above it.
It is recorded because the shape of the certificate is unusually clean and
because it separates two things the field routinely conflates.

Interfacial flow at high surface tension and low viscosity fails in two
different ways. One is a stability limit, the capillary timestep restriction
`dt <= sqrt(rho * dx^3 / (2 pi sigma))`, which falls as `dx^1.5` and comes from
resolving capillary waves explicitly. The other is spurious currents, which are
a CONSISTENCY error: the discrete surface tension force and the discrete
pressure gradient fail to cancel, leaving velocities of order sigma/mu that
nothing damps when the viscosity is small. They happen at any timestep. Only
the first is a stability problem, and they want different certificates.

The second is the one with a clean certificate. For a static interface with
exact curvature, a balanced-force discretisation makes the discrete pressure
gradient cancel the surface tension force IDENTICALLY. That is algebra, not
approximation, so it is checkable in exact rational arithmetic and lands
RIGOROUS. It certifies a scheme rather than a run, once rather than per query,
and it has a crisp binary answer against a documented failure mode. The
archetype is not the seventh: no eigenvalue is involved. It is closest to the
exact-rational verification the SOS work already uses.

What makes it worth stating separately is that exact balance holds only if the
curvature is exact, so the certificate isolates the half that can be certified
and names the half that cannot. The literature supports that split rather than
merely permitting it. Popinet and Zaleski reduced spurious currents to machine
precision in two dimensions in 1999, machine zero being direct evidence that
the pressure jump balances the surface tension force. The direct extension of
that integral formulation to three dimensions stayed open for about
twenty-five years and appears to have landed only in July 2026, in Gennari and
van Wachem, which reports spurious currents comparable to continuous surface
force methods rather than at machine precision. That is what the split
predicts: in two dimensions a marker chain gives curvature essentially exactly,
while in three dimensions curvature comes from a surface mesh or height
functions and is second order, so the residual is curvature error rather than
force imbalance. Meanwhile the route that actually put surface tension into
production in three dimensions went around the problem, through balanced-force
volume of fluid with height-function curvature, which is what Gerris and then
Basilisk do.

The capillary timestep limit is the seventh archetype's natural second home,
and the scope rule already says how well it would do. Clause two is satisfied,
since stability is a question about an extreme eigenvalue. Clause one is not:
near the threshold the eigenvalue of interest is close to zero while the
operator carries entries of order 1/dt, so the bound is loosest exactly where
it is wanted. Loose in the safe direction is still usable, since practitioners
run well inside the limit anyway, but the niche is narrow: for constant
coefficients von Neumann analysis gives the exact amplification factor and the
enclosure adds nothing. It would earn its place only with variable density and
curvature, where von Neumann does not apply.

Three things argue against entering this as a domain rather than as a tool.
Criterion four is weak, since Basilisk is C and the Python-facing route is
Firedrake or FEniCSx with real effort rather than an existing hook. The field's
queries are functionals, breakup time and droplet size distribution and bubble
rise velocity, and a stability certificate is a precondition rather than an
answer, though Kantorovich is precedent for certifying preconditions. And the
functional may not exist: this library has already met that here, since the sea
wall's peak impact pressure showed no asymptotic range and the GCI certifier
refused it. Breakup and coalescence at low viscosity are strong candidates for
the same refusal, which is a real result but not the one anyone wants.

## Recurring certificate archetypes

Seven patterns cover nearly everything above, and the rewrite library should be
organized around them.

1. **Small-parameter expansions.** Gyrokinetics, EFT, post-Newtonian,
   Braginskii. Asymptotic tier.
2. **Variational sandwiches.** Chemistry's upper and lower bracket, SOS
   transport bounds, reduced-basis a posteriori estimators. Rigorous tier.
3. **Resolution-limited queries.** HLT smearing, g−2 windows, GW mismatch. The
   ε is supplied by the instrument.
4. **Projection with memory.** Mori-Zwanzig in MD, plasma closures, subgrid
   models. Empirical tier, with variational scores where they exist.
5. **Cone-preserving brackets.** Reactor criticality, S_N transport. Rigorous
   tier.
6. **Existence certificates.** The pn junction's Newton-Kantorovich enclosure.
   Rigorous tier.
7. **Spectral enclosures.** Gershgorin, Bendixson-Hirsch, Howard's semicircle.
   A region containing every eigenvalue, from the coefficients alone, for
   operators with neither symmetry nor a cone. Rigorous tier, and loose.

A compression contributed under one archetype in one field should transfer to
its siblings mechanically. That transfer is the flywheel working. It is also
falsifiable, and the seventh archetype's first attempted transfer failed, which
is recorded below rather than quietly dropped.

### The sixth archetype

The sixth is the only one that certifies something other than a number. Its
output is the statement that a solution of a nonlinear system exists, is
locally unique, and lies within a stated radius of the computed iterate. No
residual, however small, supports that claim on its own. Its siblings are every
nonlinear steady state a simulation reports as if it were a fact: chemical
equilibrium, structural post-buckling, periodic orbits, the operating points of
a circuit.

It shipped with one restriction. ‖J⁻¹‖ came from the M-matrix witness, so the
Jacobian had to be a Z-matrix. That restriction is now lifted, by the general
route of bounding ‖I − RJ‖ for an approximate inverse R, so that the Neumann
series for (RJ)⁻¹ converges and ‖J⁻¹‖ ≤ ‖R‖/(1 − ‖I − RJ‖). Nothing was traded
away for it. Measured on twelve junction Jacobians the two routes agree to
every digit printed, which is unsurprising once you notice that on an M-matrix
the witness is *exact* rather than merely valid, because J⁻¹ ≥ 0 makes
‖J⁻¹‖∞ literally ‖J⁻¹e‖∞. The cone route is still tried first, being a solve
rather than an inverse.

The lift immediately let the archetype transfer into a launch domain and remove
a refusal there. A tokamak's pressure peaks on the magnetic axis and falls away
outward, so a real profile is *decreasing* in ψ. That does two independent
things to the Grad-Shafranov problem.

The contraction factor the existing Prager-Synge certificate needs is
max|S′|·R_max/(R_min·λ₁). For an exponential profile of decay scale ψ₀ that
maximum is c/ψ₀ rather than c, so the wall arrives ψ₀ times sooner. Measured,
it refuses above c = 0.47, where a linear source reached 2.34.

Separately, differentiating the source contributes −S′ times the *consistent
mass matrix*, whose off-diagonals are integrals of products of non-negative
basis functions and are therefore positive. A decreasing profile puts them into
the Jacobian and closes the cone route at every c > 0, including couplings the
contraction would have accepted. So the Z-matrix lift was not optional here,
and it was not the coupling strength that demanded it.

Kantorovich by the Neumann route certifies out to c = 16, a contraction factor
of 32. What changes with the method is the guarantee. Prager-Synge measures the
distance to the continuum solution, and this measures the distance to the
discrete equilibrium. The mesh gap is printed rather than folded in. It is
second order, and twelve orders above the radius a converged rung certifies.

### The continuum step, shared by three domains

That gap is now closed, and closing it in three domains at once turned out to
be one function rather than three. The reactor, the junction and the tokamak
all cross the same line, from a certificate about a discretised model to one
about the thing modelled. They now cross it through `continuum_limit`. That
function reads the ladder with `gci_extrapolate`, adds the finest rung's own
error, and takes the weaker tier. It refuses when the rungs' brackets are
within a factor of ten of the gaps between them, because a convergence order
fitted to bracket noise is fiction.

The rule that falls out is now stated once instead of three times: **a
continuum answer is never RIGOROUS, however rigorous the rungs it is built
from.** It is also usually the answer that was wanted, which is the
uncomfortable half. The tokamak's discrete radius is 10⁻¹⁴ and its mesh gap is
10⁻⁴, so the tier labels rank the two claims in the opposite order to their
usefulness.

One thing was measured and deliberately not built. Splitting a single tolerance
between the mesh and the Newton iteration looks like the obvious fifth composed
plan, and it derives nothing, because the Newton half is free. At N=200 the
junction's certificate goes from 0.78 nC/cm², which is thirty-three times too
coarse to read the mesh trend, to 1.1·10⁻⁵ in one step. That is two thousand
times tighter than the ladder needs. Every budget in the range that matters
picks the same rung, so there is no exchange rate to price, and a stage split
would be decoration on a knob that does not respond. Quadratic convergence is
what does it.

### The fifth archetype

The fifth arrived last and is the one with the most unclaimed siblings. Where
an operator preserves a cone rather than a quadratic form, Perron-Frobenius
supplies the dominant eigenvalue and Collatz-Wielandt sandwiches it from any
strictly positive trial vector, with no symmetry required anywhere. Examples
are a population that cannot go negative, or a probability that cannot go
negative. The variational archetype cannot reach any of these, because none of
the operators is self-adjoint.

The transfer claim has been tested once rather than merely asserted. The same
witness and the same bracket, with no new proof and no new code, run on an S_N
discrete-ordinates transport operator as well as on the diffusion operator they
were written for. Step differencing keeps it inside the cone, which is the only
hypothesis either function cares about. Still unclaimed, and expected to go the
same way: Markov-state relaxation rates and transfer-operator spectra in
molecular kinetics.

### The seventh archetype

The shear-flow probe above found a pattern none of the six covers, and it is
worth naming before it is built rather than after. Where an operator is neither
self-adjoint nor cone-preserving, an integral identity can still confine its
eigenvalues to a region of the complex plane, from the coefficients alone and
with no discretisation anywhere. Howard's semicircle is the instance measured;
Gershgorin and Bendixson-Hirsch are the same shape on a matrix, and Bauer-Fike
is its perturbation cousin.

What distinguishes it from the variational archetype is not just the missing
symmetry. A variational sandwich gets tighter as the trial space improves and
converges on the answer. This does not converge on anything, because it never
looked at a trial function: it is a fixed region, correct at zero cost, and
about a factor of two too large where it matters. That makes it a first rung
rather than a ladder, and the natural composition is with an untrusted
eigensolve on top of it, which is a shape this library already has.

It is now built, as `spectral_abscissa_bracket`. The upper bound is the best of
Gershgorin by rows, Gershgorin by columns, and Bendixson's
lambda_max((A + A*)/2), each on the raw matrix and on the balanced one, since
diagonal similarity preserves the spectrum and every candidate is separately
valid. The lower bound is the counting theorem: a Gershgorin component of k
discs holds exactly k eigenvalues, so every component holds at least one and
its leftmost point bounds some eigenvalue from below. RIGOROUS tier, fail_p
zero, no eigensolve anywhere. Containment holds 400 out of 400 across well
conditioned, badly scaled, strongly non-normal and complex families.

The two sides are not equally good and the asymmetry decides what the tool is
for. Median gaps over 600 matrices: the upper bound sits 0.68 above the true
abscissa on well-conditioned matrices and 0.72 on badly scaled ones, where
balancing does the work; the lower sits 7.4 and 7.7 below, because the discs
usually merge into one component whose leftmost point is far away. On strongly
non-normal matrices both degrade, to 38 and 274. So it is an upper bound with a
lower bound attached, and the query it answers is the SIGN of the abscissa.

One bug is worth recording because only the gate caught it. Building Gershgorin
components from discs of radius min(row, col) broke containment on 6 of 200
matrices, all on the lower bound. Row discs cover the spectrum and column discs
cover the spectrum; the hybrid covers neither, so the counting theorem does not
apply to it. The upper bound, which never mixes families, was never wrong.

### The combustion transfer, measured and negative

The claim first written here was that the non-normal Jacobians which made the
combustion probe expensive were a promising sibling, since a region in the
complex plane is what a Gronwall bound throws away. That was speculation and it
is now measured to be wrong, in two independent ways.

On the branched-chain skeleton the radical pool obeys, during induction, a
linear system whose abscissa is the branching rate, and ignition is that
abscissa crossing zero. Every off-diagonal is non-negative, because a radical
only ever makes other radicals, so the block is Metzler and the FIFTH archetype
reaches it. Sweeping termination through the crossover, Collatz-Wielandt lands
on the true abscissa to five decimals at every point, while the enclosure is
0.4 to 1.7 too high and its lower bound, at about -62, says nothing at all.
Where a cone exists the fifth archetype wins outright.

And on the full Jacobian, where the majors are consumed and the off-diagonals
go to -50 so no cone exists, the enclosure gives [-71.5, 12.5] against a true
abscissa of 2.5, which decides nothing. Worse, the question is degenerate
there: mass conservation pins an eigenvalue at exactly zero, so the full
Jacobian's abscissa is 0.00000 on both sides of the crossover and the branching
mode is masked. The crossover only exists in the radical subspace, and that
subspace is Metzler.

So the seventh archetype does not rescue combustion, and does not subsume the
fifth. It is a tool of last resort: where a cone exists the cone is exact, and
where none does the enclosure is often too loose to decide. Its one measured
success remains the continuous case, the Rayleigh operator, where no cone
exists, no symmetry exists, and the bound runs about 2.3x loose at the
wavenumber that matters, which is the difference between an answer and none.

### The resonance-width transfer, also measured and also negative

The second sibling was probed next and fails too, for a reason sharper than
combustion's and more useful, because it generalises into a scope rule for the
whole archetype.

A resonance is a pole of the S-matrix at E_r - i*Gamma/2, and making it a
discrete eigenvalue needs a non-Hermitian operator: a complex absorbing
potential, or exterior complex scaling. Either way the result is complex
symmetric and preserves no cone, so the cone check that combustion failed is
passed here and the seventh archetype is genuinely the only candidate.

It still fails. On a double barrier, gated against a transmission calculation
that uses no complex arithmetic at all, since a Lorentzian peak in T(E) has
full width Gamma:

    V0    b     E_r        Gamma_true     bendixson cap    loose
    10   1.50   0.81315    8.80e-03       1.2               136x
    14   1.70   3.43835    5.23e-03       1.2               230x
    20   2.00   3.64604    8.98e-04       1.2              1337x
    28   2.40   3.81795    8.38e-04       1.2              1431x

The cap never moves, because it is not about the physics. For H = H0 - i*eta*W
the anti-Hermitian part IS the absorber, so Bendixson returns exactly
2*eta*max(W): measured at 0.1, 0.4, 1.2 and 4.0 for eta of 0.05, 0.2, 0.6 and
2.0, with the potential unchanged and irrelevant throughout.

Nor can it be escaped by weakening the absorber. The CAP has to be strong
enough to absorb the resonance, so if 2*eta fell below Gamma no eigenvalue of
that width could exist at all. The bound is therefore necessarily at least the
widths it must contain, and tight only if eta is tuned to the answer being
computed.

**The scope rule, now measured on three transfers rather than argued.** Every
bound in this archetype is set by the size of the operator's non-normal part.
So it is informative when the quantity sought is COMPARABLE to that size, and
useless when it is exponentially smaller. A shear layer's growth rate and its
velocity range are both O(dU), which is why Howard runs 2.3x loose and answers
the question. A resonance width is exponentially small in the barrier while the
absorber is O(1), which is why the same bound runs 136x loose and worsens
without limit as the resonance narrows. And where a cone exists the fifth
archetype is exact and this is merely valid.

That rule is the durable product of both failed transfers, and it is why no
imaginary-axis version of the enclosure was built. The measurement says there
is nothing to point it at.

### The transfer-operator sibling, closed, and a second clause

The last sibling fails too, and for a third distinct reason that is provable
rather than merely measured, which makes it the most useful of the three.

The cone check passes trivially: a transfer operator is a stochastic matrix,
non-negative with unit row sums, so Perron-Frobenius is at home there. But the
query is not the dominant eigenvalue. Metastability is a spectral GAP, so what
molecular kinetics wants is lambda_2, and Collatz-Wielandt bounds lambda_1,
which for a stochastic matrix is 1 by construction and carries no information.

The enclosure cannot supply lambda_2 either, and the reason is an identity. For
a row-stochastic matrix the Gershgorin disc of row i has centre T_ii and radius
1 - T_ii, so its right edge is exactly 1 and its distance to the point 1 is
exactly zero. Every disc touches the Perron root. Measured, the distance came
out 0.00e+00 for every row, and the enclosure's upper bound is 1.000000 while
the true gap runs from 2e-1 down to 2e-4:

    eps      lambda_1   lambda_2    gap        enclosure upper
    1e-1     1.000000   0.800000    2.0e-01    1.000000
    1e-2     1.000000   0.980000    2.0e-02    1.000000
    1e-3     1.000000   0.998000    2.0e-03    1.000000
    1e-4     1.000000   0.999800    2.0e-04    1.000000

The bound never moves, because it cannot. The same holds for the continuous
generator, where every disc's right edge is exactly 0.

What does reach it is the SECOND archetype, which is what this document already
said the field uses: a reversible chain is self-adjoint in the stationary inner
product, so lambda_2 has a variational characterisation and any trial vector
orthogonal to the stationary one gives a rigorous lower bound. Measured on the
same chains from random trial vectors alone, 0.781, 0.932, 0.985 and 0.988
against true values of 0.800, 0.980, 0.998 and 0.9998, certifying a gap in
every case. VAC and VAMP are that argument, industrialised.

**The second clause of the scope rule.** The archetype bounds the EXTREMES of
the spectrum as a set. Any query about a subdominant or interior eigenvalue is
outside it, however non-normal the operator, because a region containing the
whole spectrum cannot separate one eigenvalue from another.

So all three siblings named for the seventh archetype are now closed, and none
transferred: combustion because a cone exists and is exact, resonance widths
because the sought quantity is exponentially smaller than the non-normal part,
transfer operators because the sought eigenvalue is not an extreme one. The
archetype has exactly one home, the Rayleigh operator, and two rules that say
why. That is a smaller claim than the one this section opened with, and it is
the one the measurements support.

## Engines, and the guess and check line

Criterion 4 says the compiler targets existing engines. The SOS work fixed the
shape of that relationship. cvxpy and SCS search, exact rational arithmetic
decides, and deleting the solver kills the search without weakening any
certificate. Every integration below sits on the same line. Engines propose.
Certificates are issued on our side of the line, and refusal stays available
whenever an engine's output cannot be checked at the declared tolerance.

**FEniCSx, the first bridge.** For elliptic problems, equilibrated-flux a
posteriori estimates (the Ern-Vohralík lineage) give bounds on the
discretization error that are guaranteed rather than estimated, and computable
from the discrete solution alone. Goal-oriented variants bound functionals.
That is a RIGOROUS-tier certificate for PDE functionals, which no resolution
ladder over a forward solver can provide at any budget. First artifact: a
certified Poisson functional with a guaranteed bound, verified on our side from
the fluxes FEniCSx returns. Then low-frequency Helmholtz, which connects back
to the wave-scattering column.

**Basilisk, the second bridge.** This is the nearest culture to this project:
adaptive refinement against a declared per-field tolerance, and a test suite of
literate pages that measure convergence orders in the open. It stops one step
short of certificates, because orders are reported and never enforced, and
there is no refusal. The bridge is to drive Basilisk as an untrusted external
binary, ladder its maximum refinement level, and mint grid-convergence
certificates on functionals. The level must be fixed per rung, because
Richardson over freely adapting meshes is not a ladder. First artifact: the
sea-wall triage rerun with the Saint-Venant solver in place of the toy WCSPH.
Second-order ladders should tighten the impulse certificate by an order of
magnitude, and may flip the low-berm refusal to a certificate. The cost is a C
toolchain (qcc), so recorded-run pages rather than CI regeneration, at least at
first.

**Reactor codes, and why not the obvious one.** The criticality checker takes
the loss and fission operators as plain arrays and asks four questions of them,
so the integration surface is already open. Any deterministic code willing to
export its assembled operators gets a certificate, with the discretization
staying that code's claim and the bracket staying ours. The obvious name,
OpenMC, is the wrong first bridge for exactly this reason. A continuous-energy
Monte Carlo code has no operator to hand over, and its k_eff arrives as a batch
standard deviation, which is a statistical statement of a different kind.
Bracketing that would mean wrapping its statistics at the EMPIRICAL tier rather
than applying the theorem. It should be built as that, honestly labelled,
rather than dressed up as the rigorous bracket beside it. A deterministic
transport code is the nearer bridge.

**Proposers per domain.** PySCF supplies integrals and reference wavefunctions
at scale while the Cholesky floors and window machinery stay ours. gwsurrogate
and bilby generate and consume waveforms around the conformal mismatch
certificate. QuTiP, stim and quimb are the engines the quantum dispatch
procedure chooses between. OpenMM, deeptime and pymbar carry molecular
kinetics, where MBAR's asymptotic variances pair naturally with the conformal
layer. Dedalus hosts the SOS transport program at PDE scale. pyrokinetics and
DESC sit under the plasma hierarchy dispatch.

**The checker side.** A short list, trusted only after declaration. python-flint
(Arb) ball arithmetic would harden the interval type and make degree-six
rational SOS cheap. mpmath already referees the test suite. The long game for
the ASYMPTOTIC tier's "exponent proven" clause is a proof assistant, and it can
wait.

Order of work: FEniCSx, then Basilisk, then PySCF. Rigor first, then the most
visible tightening, then scale.

## Sequencing

| Target | Roadmap phases | Dominant contraction | Certificate outlook |
|---|---|---|---|
| Wave scattering & CEM | 1 | rank structure, butterflies, reduced bases | rigorous / asymptotic |
| Plasmas & fusion | 2-4 | small-parameter hierarchy, closures | asymptotic to empirical |
| Quantum chemistry & materials | 2-3 | nearsightedness, low-rank ERIs, variational sandwich | rigorous intervals + declared-empirical |
| Lattice field theory | 2-3 | hierarchy, factorization, smearing | statistical + rigorous parts |
| Nuclear & dense matter | 2-3 | EFT expansion, reduced bases | statistical / asymptotic |
| Molecular kinetics & free energy | 4 | spectral-gap projection, Mori-Zwanzig | variational + declared-empirical |
| Quantum dynamics & devices | 2-4 | light cones, noise truncation, stabilizer rank | rigorous islands to empirical |
| Turbulence & transport | 4 | SOS bounds, subgrid closures | rigorous bounds + declared-empirical |
| Gravitational waves | 3 | ROM manifolds, PN/EOB expansions | empirical + asymptotic |

Beachhead order. CEM first, as the Phase 1 shakedown against MLFMA and
butterfly baselines. Then the chemistry energy bracket, as the first headline
rigorous interval. Then lattice windows and the plasma reduction hierarchy as
Phase 2 and 3 partners, chosen because both communities already budget error.
Then the quantum dispatch procedure and SOS transport bounds as certificate
showpieces. Molecular and plasma closure discovery come last, carrying Phase
4's declared-empirical banner. The selection logic throughout is to go where ε
is already spoken aloud, so the compiler's certificates land in a culture that
knows how to read them.

*Every phase of the roadmap should have a paying customer.*
