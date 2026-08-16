# MECE4520

This repository contains the materials for Columbia MECE4520:
Data Science for Mechanical Systems.

## Fall 2026 redesign goals

1. Make examples substantially more mechanical-engineering-specific.
2. Prefer real-world datasets over synthetic datasets.
3. Reuse a small number of datasets throughout the course so students
   develop familiarity with the systems.
4. Reduce introductory statistics coverage.
5. Expand reinforcement learning coverage.
6. Build a student-facing interactive course website using Quarto.

## Dataset direction

Primary recurring dataset:
- Gas Turbine CO and NOx Emissions

Secondary recurring dataset:
- CNC milling / tool wear dataset

Prefer:
- real measurements
- simple physical interpretation
- relatively small datasets
- minimal preprocessing for students

Avoid replacing every lecture with a different dataset.

## Website direction

Use Quarto.
- Keep Jupyter notebooks where computation is central.
- Use .qmd for exposition where appropriate.
- Deploy through GitHub Pages / GitHub Actions.
- Add browser-side interactivity selectively.
- Consider Shinylive/Pyodide for small interactive Python demos.
- Use Colab for computationally heavy exercises.

Initial website pilot:
- homepage
- regression notebook
- PCA/clustering notebook
- reinforcement-learning notebook

## Website implementation

- The Quarto project lives in `site/`.
- Edit `.qmd`, `_quarto.yml`, and theme/source assets in `site/`.
- Do not commit rendered `site/_site/` output or Quarto cache files.
- The GitHub Actions workflow in `.github/workflows/publish.yml` publishes
  `site/` from `master` to GitHub Pages.
- Keep the first public version focused on the homepage, grading policy, and
  syllabus; add notebooks after the pilot structure has been evaluated.

Do not reorganize the entire repository until the pilot has been evaluated.
