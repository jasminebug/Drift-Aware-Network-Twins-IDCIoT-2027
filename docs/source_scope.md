# Standalone source scope

This release is a self-contained distribution prepared from the reviewed public lightweight companion, with the same essential simulator, conditional LSTM model, drift-gate logic, and diagnostic outputs. Modules have been repackaged for a fresh repository and smoke-tested here.

**It is not the complete original training pipeline behind every number in the accepted manuscript.** In particular, the paper reports full adversarial and energy-score ablations, a separately described PPO routing study, additional synthetic scenario behavior, and two latent-rank comparisons; the provided compact executable pipeline covers only a subset. A PPO *sanity task* is not a DRL routing evaluation. Scenario S5 in this code does not simulate the manuscript's stated dual-link failure. The longer diagnostic configuration is not proof of full study reproduction.

The reported Table I/II numbers in `paper/reported/` were transcribed from the supplied accepted-author manuscript. They are labeled reported rather than generated. No fabricated experiment logs, derived data for Figure 3, or fabricated significance/error bars have been added. Please release original full-experiment scripts/logs separately if available.

The included paper PDF is the author's IEEE-formatted version provided in this conversation, not proof of final proceedings publication.
