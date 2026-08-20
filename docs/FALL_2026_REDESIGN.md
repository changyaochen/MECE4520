# Fall 2026 redesign

## Purpose

This branch is the working area for a redesigned MECE 4520 course site for Fall 2026. It begins from the `fall_2026` snapshot branch so that the former course materials remain available for reference while the new experience is developed independently.

## Starting state

- The legacy `lectures/` directory has been removed from this branch.
- Existing repository content outside `lectures/` remains in place unless deliberately revised.
- The current `README.md` reflects Fall 2025 information and should be treated as source material, not Fall 2026 published copy.

## Design principles

1. Make the course’s essential information easy to find: schedule, staff, policies, assignments, and resources.
2. Organize instructional materials around the Fall 2026 learning experience rather than the legacy directory layout.
3. Keep the site approachable for students new to data science while preserving technical precision.
4. Use accessible, durable formats and links that work without local course-specific setup.

## Open decisions

- Information architecture and navigation for the new site.
- The home page’s Fall 2026 course information, syllabus, and key dates.
- The format and location of new learning materials.
- The deployment and publishing workflow.

## Asynchronous foundations pilot

The September 8 class will be asynchronous. Its first implementation is a
Quarto landing page with three short, standalone Jupyter notebooks:

1. Course tools and Python foundations.
2. A first look at the gas-turbine emissions data.
3. Probability distributions and simulation.

The notebooks deliberately use the same gas-turbine dataset as the later
regression material. They can be read as part of the Quarto site or run in
Google Colab. Each carries its own setup cell so a student can begin from a
fresh Colab runtime. The third notebook introduces NumPy-based simulation and
distribution shapes as preparation for the September 10 in-class CLT and
hypothesis-testing discussion. Sampling distributions, bootstrap intervals,
and p-values are deliberately taught with instructor guidance rather than
assigned for asynchronous exploration.
