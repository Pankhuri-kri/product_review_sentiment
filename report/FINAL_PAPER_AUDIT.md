# Final paper audit

## Two column paper audit 30 October 2026

The current PDF has **exactly nine pages of main content on pages 3–11** in a two-column layout based on the supplied DM_project-1.pdf reference. Cover, index, references, and contribution statement remain separate: **13 pages total**.

A content-retention audit confirms that all prose is unchanged, all six tables retain their details, all five displayed equation groups are mathematically equivalent, and all eight figure paths and captions are retained. Author metadata, cover text, references, and contribution statements are byte-identical to the previous version. Long equations were broken across lines, and wide figures and tables span both columns.

All **52 paper checks** passed for this version. All **55 saved experiment-artifact hashes** matched; all **29 index destinations** match the displayed page numbers. All **13 rendered pages** were visually reviewed, including the balanced two-column reference list. Final compilation has no unresolved references, overfull boxes, or missing-character warnings.

The layout and content audits are recorded in `verification/paper-template/final-paper-audit.json` and `verification/two-column-paper/retained-content-audit.json`. No training outputs were changed.

## Earlier single column expansion 30 October 2026

The expanded research paper has **15 total pages**. **Main content spans pages 3–13, giving 11 pages excluding the cover, index, references, and contribution statement.** References start on a separate page, making the main-content count unambiguous. Every main-content page has more than 400 extracted words including its tables and figure labels.

The revision adds substantive explanation of the research questions, representation and padding differences, model computations, selection controls, sequence coverage, class-specific behavior, shared errors, resource accounting, threats to validity, and controlled follow-up experiments. The original experiment outputs and reported model scores are retained.

Eight vector figures and six tables are included. New figures are generated from the saved final experiment using the accompanying `figure_data.json` and `generate_figures.py`. The storage plot distinguishes recorded sparse float32 storage from calculated dense float32 and float64 baselines. Stage durations are distinguished from internal fitting durations, and exact error-set membership is distinguished from overlapping heuristic linguistic cues.

All **52 paper checks** passed, including independent reconstruction of metrics and figure confusion matrices from the 2,000 predictions per model. All **55 recorded experiment-artifact hashes** matched. All **29 clickable index destinations** match their displayed page numbers. Compilation completed with resolved citations and references and no overfull boxes or missing-character warnings. All **15 rendered pages** were visually inspected.

The detailed paper audit is recorded in `verification/paper-template/final-paper-audit.json`. Members must still review their contribution statements and supply signatures.

## Earlier audit 27 October 2026

The following records the earlier ten-page version before the present expansion.

Reviewed on 27 October 2026 against the supplied report template, Phase 1 synopsis, Phase 2 interim report, executed source code, and saved Amazon Polarity experiment artifacts.

The paper is consistent with the documented project scope and the actual final experiment. All 44 paper checks and all 38 independent experiment-result checks passed. All 55 recorded experiment-file hashes matched. The final PDF compiled successfully and all ten rendered pages were visually inspected.

## Template and earlier reports

- The title, names, registration numbers, three model assignments, dataset scope, and three objectives agree with Phase 1 and Phase 2.
- The paper follows Sections I–VIII in the required order, with an unnumbered acknowledgement, references, and a separate contribution page.
- The abstract contains 199 words and the keyword list contains eight entries.
- The prose literature review includes all ten interim-review works. The bibliography contains 15 consecutively numbered references in first-citation order. Joulin's 2016 preprint is cited using its 2017 conference publication.
- A4 layout, one-column body, Times New Roman prose, title/author/body/email font sizes, heading hierarchy, caption positions, and table/figure numbering were checked against the template and rendered output.
- The index and centered page-number footers are intentional changes explicitly requested by the user. The original template specifies no manuscript page numbers. All 26 clickable index entries agree with their destination pages and the displayed page numbers.
- The Phase 2 contribution log is retained: Pankhuri owns acquisition/TF-IDF/MLP, Arya owns cleaning/tokenization/BiLSTM, and Sai owns splitting/TextCNN/evaluation/comparison code. Final responsibilities are extended according to those model assignments.

## Actual results

Metrics were independently recomputed from the 2,000 saved test predictions per model. Precision, recall, and F1 refer to the positive class.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MLP on TF-IDF | 88.20% | 0.8760 | 0.8900 | 0.8829 |
| TextCNN | 85.40% | 0.8371 | 0.8790 | 0.8576 |
| BiLSTM | 86.10% | 0.8588 | 0.8640 | 0.8614 |

Verified evidence includes:

- A real, balanced 20,000-review subset with 16,000/2,000/2,000 stratified partitions, seed 42, and no cross-partition review-ID or cleaned-text overlap.
- Training-only fitting of lexical/token representations; shared sequence inputs; the pinned dataset revision; unchanged executed source hashes.
- Two candidates per model, validation-only selection, eight-epoch maximum, patience two, batch size 32, fixed architecture sizes, and the actual selected hyperparameters.
- Parameter counts of 5,136,769 / 2,072,449 / 2,092,801; selected best epochs of 1 / 2 / 1; training histories and validation losses.
- Error counts of 236 / 292 / 278; 103 errors shared by all models; 470 errors in their union; nine reviewed source cases with all three predictions verified.
- Heuristic cue counts of 1,352 / 220 / 18, described as overlapping heuristic subsets rather than linguistic ground truth.
- 593 truncated test reviews (29.65%); sparse feature storage; maximum recorded worker-process-tree private memory of 686.5 MiB; 818.78 seconds summed across 12 isolated stages.
- Figure loss trajectories visually cross-checked against the selected-run CSV histories. The figure and test-result table describe the final run.

The official Amazon test partition was not used. The paper correctly identifies its test set as a held-out portion of the sampled source-training partition. It makes no multi-seed, statistical-significance, causal-feature-attribution, or universal model-superiority claim.

## Clarifications added during this audit

Section V now explicitly distinguishes the final audited run from the preliminary five-epoch Phase 2 runs. It explains the actual two-pair dropout/learning-rate search, fixed embedding/filter/unit settings, and local CPU execution instead of the envisaged Colab environment. Changes in scores between phases are not presented as a controlled ablation of regularization. The abstract's memory description was aligned with the recorded process-tree measurement.

The preliminary Phase 2 accuracies (86.55%, 84.85%, 82.55%) belong to the earlier experiment and are not final result targets. Matching the reports means preserving their scope and documented development history while reporting the actual final measurements. The reports alone do not establish identical review IDs between the interim and final runs.

## Human completion

Each member still needs to review their drafted contribution statement and sign the contribution page. The task log and model assignments support the statements; automated artifact checks cannot establish individual human authorship. Signatures have intentionally been left blank.

## Reproduce the audit

From the project workspace, use the existing verified experiment environment for `verification/audit_fixed_results.py` and the bundled artifact Python runtime for `verification/paper-template/audit_final_paper.py`. These audits read the saved experiment and paper; they do not retrain or alter model outputs. Detailed checks are saved in `verification/fixed-results-audit.json` and `verification/paper-template/final-paper-audit.json`.
