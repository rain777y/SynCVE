# Evaluation Reports

This directory stores curated evaluation summaries and selected structured run records. It is the main source for evidence that may later be reused in demos, papers, or slide decks.

## Report Types

| Type | Examples | Use |
| --- | --- | --- |
| Narrative summaries | `system_evaluation_wave12.md`, `ablation_findings.md`, `phase2_analysis.md` | Explain what was tested, what improved, and what remains open. |
| Demo outputs | `demo_clinical_report.md`, `demo_session_summary.json` | Show end-to-end product behavior and report generation. |
| Structured experiment records | `full_ablation.json`, `flood_e2e_*.json` | Preserve machine-readable evidence from evaluation or stress runs. |
| Handoff context | `handoff_context.md`, `optimization_decision.md` | Capture decisions and next-step context for collaborators. |

## Curation Rules

- Put readable summaries here when they support industrial validation, academic framing, or project handoff.
- Keep raw caches, downloaded datasets, and intermediate generated files outside this directory.
- Prefer stable filenames for reusable reports and timestamped filenames for individual runs.
- When a result is used in external communication, cite the exact report or JSON record that supports it.

## Current External-Ready Entries

- `demo_clinical_report.md`: sample clinical-style report output.
- `demo_session_summary.json`: structured companion data for the demo report.
- `system_evaluation_wave12.md`: current consolidated evaluation narrative.
- `ablation_findings.md`: ablation-level evidence for method discussion.
- `full_ablation.json`: machine-readable ablation result record.
- `flood_e2e_1778393619.json`: selected recent flood test record for operational behavior.
