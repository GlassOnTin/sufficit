# Learning under a budget

**What an agent would be if it were built the way sufficit is built.** Not a certified physics library with a language model bolted to it, but a reasoner whose foundations are the same three: an untrusted proposer, a cheap checker, and a tier on every claim.

> **Status:** an argument, offered for the reason [VISION.md](VISION.md) was offered — to be argued with. Sufficit is a working existence proof for one half of what follows and contributes nothing at all to the other half. This document is careful throughout about which half is which, because the interesting claim is small and the uninteresting version of it is very easy to write.

## The problem

An agent that reasons today produces one artifact: prose, at one confidence, at whatever effort it happened to spend. Ask a harder question and it answers in the same register, at the same length, with the same air of having finished. Nothing in the output separates a step that was checked from one that was fitted from one that was invented, and nothing in the input let you say how wrong the answer was allowed to be. The error is not bounded, not estimated, and not declared; it is simply absent from the type.

That is exactly the situation sufficit was built to fix for physics, where the pathology was easier to see because the field already had units for it. A reactor code stops its iteration when k stops moving. A device simulator stops Newton's method when the residual looks small. Both are hopes, and both are hopes of a specific measurable size: on the [pn junction](https://glassontin.github.io/sufficit/cases/junction.html), a residual of 5·10⁻⁶ — a number any engineer would call converged — certifies the answer only to half a per cent, because ‖J⁻¹‖ is 237 and the residual shows none of that. The gap between what a stopping rule reports and what it has established is not a rounding detail. It is the whole quantity of interest.

## The thesis

**The transferable idea is not the certificates. It is the asymmetry underneath them.**

In sufficit the proposer is untrusted and the checker is cheap. Optimizers, SDP solvers, sketches, warm starts and outright bad guesses all propose; the certificate is issued on the other side of a line, from hypotheses checked at runtime. A bad guess costs tightness, never truth. That single property is what lets an arbitrarily unreliable generator sit at the front of the pipeline with nothing at risk but time.

Point that at a language model and the consequence is immediate: make it a proposer whose job is to make the checker's search short. Inside the checkable fragment, hallucination stops being a truth problem and becomes a cost problem. It also inverts the training target. You would not train the thing to be right. You would train it to be cheap, and check it for rightness — and you would measure it in certified answers per unit compute at a stated tolerance, which is a number nobody currently reports.

The second half of the thesis is the tier discipline. Every claim carries a value, an error, a tier and a provenance chain, and composition keeps the weakest tier. That is worth more than it sounds, because the characteristic failure of current systems is not being wrong. It is flattening a proved step, a fitted extrapolation and a guess into one confident paragraph. A tiered representation makes that distinction mechanical instead of rhetorical, and it does not demand rigour everywhere: sufficit's own EMPIRICAL tier is conformal prediction, which needs only exchangeability, and exchangeability is available to a general learner in a way that variational principles are not.

## The three kinds of learning

The part of this worth building a system around is that *learning* is not one activity. It splits into three, with sharply different risk, and keeping them apart is the trick.

**1. Learning what things cost. Safe, continuous, and it should be aggressive.** Every rung the planner runs logs its predicted cost against its measured cost, in the receipt attached to the winning certificate and in the `tried` field of every refusal. A wrong cost model buys extra attempts and cannot corrupt an answer, because cost models order the attempts and certificates arbitrate. So this learning can run hard precisely because being wrong is only expensive. Compare a learned reward model, where an error propagates straight into the policy. Sufficit already lives off this: measuring that ten correction iterations buy 94–101% of what eighty buy took an 8-tolerance sweep from 86.5 s to 4.9 s with no certified result lost.

**2. Learning where a technique applies. Cheap, and the source of all transfer.** Index a technique on its *hypotheses*, not on its domain. The concrete case is in this repository. `mmatrix_witness` was written to prove that a reactor's loss operator has a non-negative inverse. It now supplies the Kantorovich constant for a semiconductor's nonlinear Poisson equation, with no new proof and no new code, because what the function requires is "Z-matrix", not "neutrons". The physics has nothing in common; the cone is the same.

That transfer also bought something reuse alone does not. It broke. The witness had a hardcoded 10⁻⁹ of slack, which silently assumes the linear solve behind it was accurate to better than that. True for a reactor. False at condition number 4·10⁸, where the solve is wrong in the eighth digit and the witness refused a perfectly good matrix. **Moving a technique somewhere it nearly breaks is a learning signal you cannot get by using it where it already works** — and the repair, scaling by the measured shortfall instead of a constant, is strictly better than what was there before.

**3. Learning new certificates. Unsolved, and sufficit does not do it.** This has to be said plainly, because it is where an enthusiastic reading of this project goes wrong. Sufficit's rewrites are proved by hand, one per problem, and the module that holds them plus its planner runs to about 5,900 lines across eleven domains. The compiler half searches over pieces that were already proved; it invents none of them. Sufficit is an existence proof for economical *planning*, not for learning from the ground up.

The automatable fragment of certificate discovery is narrower than it first looks, and it is where effort should go: composing certified pieces under one budget, which is what `compose` does over a graph of stages; and relaxing hypotheses to find the wider class where a proof still holds, which is what turned a reactor's witness into a semiconductor's.

## Two things that are cheap and that nobody does

**Refusal that is priced.** Sufficit refuses constantly, and every refusal carries the receipt: the rungs tried, their measured errors and costs, and what the next move would cost. That is what makes a refusal something you can plan around rather than a dead end. An agent that says "I can get you within 5% now, or within 0.5% for roughly a hundred times the compute" has told you the shape of what you did not get.

**Measured negatives, kept with their numbers.** Before entering combustion this project measured that the spectral abscissa integrates to ∫α dt ≈ 17 over the induction period of a hydrogen mechanism, so perturbations are amplified 2.5·10⁷ times as a matter of physics, and any forward error bound pays at least that. A fixed-norm bound pays far more: ∫μ dt ≈ 3000 in the natural scaling, ≈ 67 after optimal diagonal balancing, because chemical Jacobians are severely non-normal. The domain was not entered, and the reason is a permanent, reusable, quantitative result rather than a note that it seemed hard. The same happened inside the planner: escalating the binding stage instead of walking the cost ladder was built, measured at 5 against 6, then 6 against 6, then 7 against 6 in the thing that actually costs, and removed. Most agents forget their dead ends inside a session, or record them as prose that reads like an excuse.

## Honest limits

**Coverage is the real constraint, and it runs the wrong way.** Sufficit gets to be mostly-certified because it chose domains where certificates exist. A general agent has the opposite mix: most of what people want is not in a checkable fragment at all, and no amount of architecture puts it there. So the achievable version of this is not "everything certified". It is **everything labelled, with the labels composing honestly**, which in practice means a great deal of EMPIRICAL with a printed failure probability. That is a much weaker claim than the one this document could have made, and it is still a large improvement on what exists.

**A tolerance needs a scalar functional.** "Within ε" is meaningful for a binding energy, a criticality eigenvalue, a depletion charge, a waveform mismatch. Choosing the functional for a question that did not come with one is a modelling act, and a badly chosen functional hides the real question behind a number that is easy to bound. This is not a technicality; it is where a system like this would most easily fool its user.

**Exchangeability is an assumption, and it is the one that breaks.** The distribution-free tier is only as good as the claim that the calibration set and the new case are drawn alike. Under distribution shift the failure probability is not 1/(n+1), and the certificate says nothing about how far off it is.

**The checker must be cheaper than the proposer, and often it is not.** The asymmetry is the whole engine. Where verifying costs as much as generating, none of this pays, and pretending otherwise produces an expensive system with the same guarantees as a cheap one.

## What success looks like

**Near term:** an agent that takes a tolerance as input and returns a tiered claim with its provenance, refuses when it cannot meet the tolerance, and prices the refusal. None of this needs a research result. The parts are conformal calibration, cheap verifiers wherever they exist, a claim type with a weakest-link composition rule, and receipts that audit the planner's own cost model.

**Mid term:** the receipt loop closes. The system's cost models measurably improve from its own logged attempts, with the correctness guarantee provably untouched by that learning, because the two are separated by construction rather than by care.

**Long term:** the flywheel turns without a human. A technique acquired for one problem is indexed by its hypotheses, is found automatically when those hypotheses hold somewhere unrelated, and its hidden assumptions are exposed by the transfer rather than by an incident. That is the step this repository has taken once, by hand, and can currently take only by hand.

## Non-goals

Not a claim that reasoning can be made rigorous. Most of it cannot, and the tier system exists to say so in the type rather than in a hedge. Not an argument against large models; the proposer is load-bearing and the better it is, the shorter the checker's search. Not a safety argument, though the structural habit transfers: the enforcement that keeps this project's combinator honest is a check in the planner and a check in the build, not an instruction to remember.

## Intellectual lineage

Vovk, Gammerman & Shafer (conformal prediction, the distribution-free tier); McConnell, Mehlhorn, Näher & Schweitzer (certifying algorithms, the proposer/checker split); Moore and Rump (interval and validated numerics); Kantorovich (existence from a residual); Perron and Frobenius, via Collatz and Wielandt (bounds from a cone rather than a quadratic form); Frigo & Johnson, Püschel et al., Ragan-Kelley et al. (specification separated from schedule, schedule space searched); Panchekha et al. and Darulova & Kuncak (error carried through rewrites by a compiler).

---

*Sufficit's bet is that the map from model to answer is compressible and that the compression can be found mechanically. The bet here is narrower and, if it holds, more useful: that the compression is worth having only when it arrives with a declared bound, and that the discipline of declaring one is portable off the physics.*
