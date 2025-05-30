# Premier League Causal Analysis


## Abstract

This project analyzes the causal impact of mid-season head coaching
changes on team performance in the 2023–2024 Premier League season.
While these changes are often made in response to poor results, it’s
unclear whether they actually lead to better outcomes. Using match data
and coaching history, the analysis controls for key factors like team
strength, opponent difficulty, and match context. By applying regression
and Bayesian methods, the goal is to estimate the true effect of
coaching changes on performance, measured in points earned. The results
aim to help clubs evaluate whether these decisions are worth the
cost—both on and off the pitch.

## Project Organization

- `/code` Scripts with prefixes (e.g., `01_import-data.py`,
  `02_clean-data.py`) and functions in `/code/src`.
- `/data` Simulated and real data, the latter not pushed.
- `/figures` PNG images and plots.
- `/output` Output from model runs, not pushed.
- `/presentations` Presentation slides.
- `/private` A catch-all folder for miscellaneous files, not pushed.
- `/writing` Reports, posts, and case studies.
- `/.venv` Hidden project library, not pushed.
- `.gitignore` Hidden Git instructions file.
- `.python-version` Hidden Python version for the reproducible
  environment.
- `requirements.txt` Information on the reproducible environment.

## Reproducible Environment

After cloning this repository, go to the project’s terminal in Positron
and run `python -m venv .venv` to create the `/.venv` project library,
followed by `pip install -r requirements.txt` to install the specified
library versions.

Whenever you install new libraries or decide to update the versions of
libraries you use, run `pip freeze > requirements.txt` to update
`requirements.txt`.

For more details on using GitHub, Quarto, etc. see [ASC
Training](https://github.com/marcdotson/asc-training).
