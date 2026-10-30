# LaTeX final research paper

## Two column research paper 30 October 2026

The compiled paper has **13 pages**, including **exactly nine pages of main content on pages 3–11**. The cover (page 1), clickable index (page 2), references (page 12), and contribution statement (page 13) are excluded from this main-content count.

The manuscript uses the reference PDF's two-column research-paper structure, with a full-width title and a clear central gutter. Wide tables and multi-panel figures span the two columns. All prose, table values, equations, figure captions, author details, references, and contribution statements are retained from the expanded report; only formatting and float placement change.

The manuscript retains the required eight-section research-paper structure, a 199-word abstract, eight keywords, 15 numbered references, six tables, and eight vector visualizations:

1. Shared experiment workflow and selection boundaries.
2. Implemented MLP, TextCNN, and BiLSTM architectures.
3. Balanced data partitions and sequence truncation.
4. Selected training-stage duration and feature storage.
5. Test accuracy, precision, recall, and F1.
6. Three class-normalized confusion matrices.
7. Selected-run training and validation loss.
8. Exact intersections of the three error sets.

The figures use the saved final experiment. `figure_data.json` contains the exported values; `generate_figures.py` rebuilds the vector PDFs using Python and ReportLab. Figure files are included, so recompiling the paper does not require running the figure generator.

The common cover from `ICT_4442_Mini_Project_Report_Template (2).docx` is retained. The manuscript uses A4 pages, two columns, 10 pt Times New Roman body text, a 24 pt title, Roman-numbered main sections, IEEE-numbered references, and a separate contribution page. It includes the template's original logo and eight vector figures from the saved experiment.

At the team's request, a clickable index (table of contents) is included after the cover. Centered footer page numbers match both its page references and the PDF viewer's page numbers, starting with page 1 on the cover.

## Build

Use **XeLaTeX** with Times New Roman and Courier New installed. From this folder:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
```

The document uses standard TeX packages: fontspec, geometry, amsmath, amssymb, graphicx, array, tabularx, booktabs, titlesec, multicol, and hyperref. The bibliography is supplied directly in `references.tex`; BibTeX/Biber is not required.

## Files

- `main.tex`: formatting and document assembly.
- `metadata.tex`: project title and author emails.
- `cover.tex`: template cover.
- `index.tex`: automatic table of contents, including sections and subsections.
- `manuscript.tex`: abstract, keywords, Sections I–VIII, and acknowledgement.
- `references.tex`: 15 numbered research references, including the ten works from the interim literature review.
- `contributions.tex`: member/model identifiers and key contributions based on the Phase 1 and Phase 2 reports, extended to each model owner's final training, tuning, evaluation, and analysis responsibilities. Signature fields remain blank.
- `figures/`: original Manipal logo and eight vector figures.
- `figure_data.json`: experiment values used in the visualizations.
- `generate_figures.py`: reproducible figure generation using ReportLab.

## Fields still to complete

All three supplied author email addresses are included in `metadata.tex`. Key contributions are filled in; signatures remain for the members to provide.

## Evidence

The paper uses the verified **single-seed** experiment under `../../results/amazon_20k/`, with test accuracies 88.20%, 85.40%, and 86.10% for MLP, TextCNN, and BiLSTM. It does not adopt unsupported three-seed averages or larger tuning grids from the separate completed Word report. The nine reviewed failure cases and 103 shared errors are drawn from the saved real-data predictions and qualitative review. No training outputs were changed.
