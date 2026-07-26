# Zero-Shot Text Classification with Locally-Hosted LLMs — Paper sources

LaTeX sources for the `ollama-classifier` paper. The bibliography uses
**`biblatex` with APA style and the `biber` backend**, so the PDF must be built
with **`biber`, not `bibtex`**.

## Requirements

- A TeX Live installation providing `pdflatex`, and the packages
  `biblatex`, `biblatex-apa`, and their dependencies.
- `biber` (its version must match `biblatex`: e.g. `biblatex` 3.x ↔ `biber` 2.x).

On Debian/Ubuntu, install everything at once:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended \
                 texlive-bibtex-extra biber
```

## Build the PDF

From the project root (`main.tex` lives here), run:

```bash
pdflatex main
biber main
pdflatex main
pdflatex main
```

The four passes are required because cross-references and citations need to
stabilise: the first `pdflatex` emits the `.bcf` control file, `biber` reads it
to produce the bibliography (`.bbl`), and the last two `pdflatex` passes resolve
the in-text citations and bibliography.

A freshly compiled `main.pdf` (18 pages) is produced in the project root.

## Rebuild from scratch

To remove all generated artifacts and recompile cleanly:

```bash
rm -f main.aux main.bbl main.bcf main.blg main.log main.out \
      main.run.xml main.synctex.gz main.toc
pdflatex main && biber main && pdflatex main && pdflatex main
```

## Common pitfall: citations show as `[?]` or the bibliography is missing

This happens when **`bibtex` is run instead of `biber`**. `bibtex` cannot read
the `biblatex` control file (`main.bcf`), so it emits an empty `main.bbl` and
all citations break. Always use `biber main`. If you see `main.bbl` is 0 bytes,
delete it and re-run the `biber` step.

## Project layout

| Path                | Contents                                              |
|---------------------|-------------------------------------------------------|
| `main.tex`          | Paper source.                                         |
| `references.bib`    | Bibliography database (read by `biber`).              |
| `experiment/`       | Benchmark scripts, data, and figures included via `\graphicspath`. |
| `main.pdf`          | Compiled output (generated).                          |
