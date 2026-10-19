# EN3150 Assignment 03 report

This directory contains everything needed to compile the report, apart from the LaTeX compiler and its standard packages. It can be copied separately from the repository.

- `report.pdf`: compiled report.
- `report.tex`: report source.
- `references.bib`: bibliography.
- `tables/`: LaTeX result tables.
- `figures/`: local copies of every figure used by the report.

## Build

From this directory, run:

```sh
tectonic report.tex
```

Tectonic may download standard TeX packages on its first build. Original experiment data and checkpoints remain in the project; they are not needed to compile this report. Local figure copies are snapshots of the verified saved results, not regenerated measurements.

Before submission, complete the group number, remaining member names/index numbers, and submission date in `report.tex`, and review the report as a group.

## Result selection

Custom-model results and figures use the completed controlled single-seed study `controlled_ab_seed42_20261019_215110`: matched 40-epoch recipes, Adam/SGD/momentum SGD. Pretrained runs remain separately labelled references. Earlier unmatched custom runs are not the reported A/B results.
