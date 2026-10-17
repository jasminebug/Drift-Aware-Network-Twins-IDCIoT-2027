# Research record update — accepted IDCIoT 2027 manuscript

**Manuscript ID:** IDCIoT-068  
**Title:** Drift-Aware Reference Shrinkage for Proactive Traffic Engineering with Generative Network Twins  
**Conference:** IDCIoT 2027 — accepted, January 21–23, 2027  
**Status:** Documentation and source-data provenance update; **not** a full experimental reproduction certification.

## Updated

- Public README, author metadata and citation to reflect the revised paper title.
- Structured copies of all three printed manuscript result tables (unavailable Table III cells remain blank).
- Source-derived bar and line charts for Tables I/II, isolated from experimental output.
- Reproducibility notes mapping the current code to the accepted paper, including unresolved S5, GAN, and PPO discrepancies.
- Math-level consistency checks for reported CRPS improvement and oracle-gap closure.

## Standalone release scope

The archive includes self-contained compact source files in `src/`, a diagnostic runner, unit tests and all required metadata. These modules follow the essential public companion approach but are **not an archive of the complete original paper experiments**. The previous repository need not exist for this archive to be uploaded or examined. The source code's diagnostic output directory is empty except for its README.

A complete audited paper-equivalent reproduction still requires the actual full-run experimental code, raw logs, configurations and any missing scenario implementations. No published result has been changed to simulate reproducibility.
