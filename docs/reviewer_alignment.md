# Accepted-paper change register — IDCIoT-068

This file documents correspondence between the **revised IEEE camera-ready manuscript** and its public code repository, not an additional review of the research.

| Topic | Present paper state | Repository treatment |
|---|---|---|
| Final title | *Drift-Aware Reference Shrinkage for Proactive Traffic Engineering with Generative Network Twins* | README, software citation and paper citation updated |
| Conference | Accepted for IDCIoT 2027, 21–23 January 2027 | Described as **accepted**, not already published in IEEE Xplore |
| Forecast calibration | Empirical coverage at nominal 90% improves from 2.8% to 74.7% | Historical **reported** Table I copied as a labelled transcription |
| CRPS change | 0.3377 to 0.2501, or 25.9% improvement | Checked via a deterministic arithmetic test |
| Proactive TE | Table II contains six scenarios and six reference strategies | Source figures redrawn from Table II, not from new simulations |
| Reviewer comment on S3 | 40.37 ms ungated versus 34.08 ms gated | Reported values and 15.6% relative difference preserved |
| In-distribution oracle gap | 39.44 ms reactive, 34.31 ms gated, 33.80 ms oracle | 90.9% gap closure basis checked |
| Under-calibration | Still 15.3 percentage points short of nominal | Explicitly acknowledged as a limitation |
| Multi-seed significance | Not established | No error bars or significance claims invented |
| Real network / second topology | Not established | No external-trace or second-topology validation claimed |
| Gate sensitivity | Parameters not validated by a documented independent sweep | Documented gap; not represented as complete experiment |
| CVaR | K=8 with a two-member upper tail | Marked as coarse risk estimate |
| Graph format | Source Figures 2, 3, 5 revised in paper | Source-derived plots for Tables I/II included; Figure 3 data unresolved |
| Bibliography | 20 references in revised paper | Final title/authors recorded; no extra references invented |

### Scientific integrity safeguard

The public code, the accepted paper, and the supplementary reported tables represent **different types of evidence**. `outputs/` contains quick-run pipeline artifacts. `paper/reported/` contains observations printed in the manuscript. Only execution logs from the actual full configuration can support a claim of numerical reproducibility.
