# Contributing to SPADE

Thank you for your interest in contributing. SPADE is an open-source research tool, and contributions of all kinds are welcome.

## Ways to Contribute

- **Bug reports** — open an Issue describing the problem, the error message, and your Python/OS version
- **Feature suggestions** — open an Issue with the `enhancement` label
- **Code contributions** — fork the repository, make your changes, and open a Pull Request
- **Documentation** — improvements to the README, docstrings, or user guide

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/spade.git
cd spade
pip install -r requirements.txt
streamlit run wheat_dashboard.py
```

## Code Style

- Follow PEP 8 for Python style
- Add docstrings to new functions explaining their purpose, inputs, and outputs
- Use descriptive variable names — agronomic abbreviations (AE-N, RE-N, NHI) are fine and preferred where established

## Pull Request Guidelines

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear commit messages
3. Test that the dashboard runs without errors: `streamlit run wheat_dashboard.py`
4. Open a Pull Request with a description of what you changed and why

## Planned Features (Good First Issues)

- RCBD (Randomised Complete Block Design) support with mixed-effects models
- Dose-response curve fitting for NUE as a function of N rate
- Effect size (η², ω²) integration into the two-way ANOVA table
- Multi-environment analysis (AMMI, GGE biplot)
- Conda environment.yml for exact dependency pinning

## Questions

Open an Issue with the `question` label or contact via [sajjadur-rahman.com](https://sajjadur-rahman.com).
