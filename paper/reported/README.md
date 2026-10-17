# Manuscript-reported numerical values

These files are **transcriptions from the author's IEEE-formatted camera-ready document**, not outputs of an independent rerun of the GitHub pipeline.

- Source: `IDCIoT-068_IEEE_Template_Formatted.docx` (held by the authors; not included here to avoid unsolicited manuscript redistribution).
- Source SHA-256: `62967b0639b55435d275ed72c299bb922dd2a8059a4d20ef6b511e08c13a8bb3`.
- **Table I:** test forecasting scores at K=100 as printed. The source fifth-column header is blank. It is intentionally named `unlabeled_fifth_column_in_manuscript`; no metric name is assumed.
- **Table II:** mean routing delay (ms) over In-d., S1–S5, with the manuscript's reported reference-matrix strategies.
- **Table III:** source cells are incomplete, including missing header labels and result cells; reproduced here strictly as printed, not filled with inferred numbers.
- These values are **reported evidence**, not a proof that the existing code can recreate them.

Use `python scripts/plot_reported_tables.py` to regenerate the two tabular illustrations. Running that command does **not** train or evaluate the forecaster or routing controller.
