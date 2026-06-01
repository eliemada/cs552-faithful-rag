# Reports

All reports use the LaTeX templates provided by the course.

## How to add templates

Download the templates from Overleaf and place them in the corresponding folders:

```
report/
├── proposal/              # 1-page proposal (due May 3)
│   └── main.tex
├── literature_review/     # 1-page lit review (due May 3)
│   └── main.tex
├── progress_report/       # 1-page progress report (due May 24)
│   └── main.tex
└── final_report/          # 4-page final report (due June 7)
    └── main.tex           # Main entry point (matches Overleaf filename)
```

## Overleaf Links

Team editable projects (drafting + compilation happen here; local `.tex`
files are kept in sync via the Overleaf MCP and git):

- Proposal + Lit Review (M1): https://www.overleaf.com/project/69de33fda50b37be7b6b83af
- Progress Report (M2): https://www.overleaf.com/project/6a0b0b0a59eaf83e286fc6fc
- ~~Final Report (M3, first attempt):~~ https://www.overleaf.com/project/6a159f09442f9f862f3ea730 _(deprecated 2026-06-01 — superseded by the project below; do not edit)_
- **Final Report (M3, active):** https://www.overleaf.com/project/6a1d901a3bed1b9e7bc6b091

Course-provided read-only templates (for reference / re-copying):

- Final Report (Open Project): https://www.overleaf.com/read/dfchdbrspjhh

## Notes on `report/final_report/`

- `anthology.bib` (~42 MB ACL Anthology references) lives in Overleaf
  only — it's `.gitignore`'d to keep this repo lean. Overleaf already
  has it; nothing to do for compilation there. If you need it locally,
  fetch it with `curl -L -o report/final_report/anthology.bib https://aclanthology.org/anthology.bib`.
- `main.pdf`, `main.bbl` are produced by Overleaf on each compile;
  ship the latest `.pdf` to the repo only when tagging a submission.
