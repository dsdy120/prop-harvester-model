# Todo
- [X] Find GMAT version that supports [NRLMSISE-00](https://en.wikipedia.org/wiki/NRLMSISE-00)
  - Doesn't seem to exist
- [ ] Complete tutorial
- [ ] Find tutorial for GMAT scripting
- [ ] Find tutorial for GMAT Python API

# 2025-03-14
- Switched to GMAT
- To use latest epoch ISS data for first-cut orbit\

# 2025-03-15
- Settled on [MSISE-90] atmospheric model since it provides better results than Jacchia-Roberts model (see https://www.sciencedirect.com/science/article/abs/pii/S0094576501000157)
- Added Lunar and Sun perturbations in case there is any effect on the orbit
- Decided on SPICE attitude model for harvester to allow maximum control by Python script

# 2025-03-21
- branching to try direct Python
- never mind, GMAT is better for now
- Found sample SPAD file for JWST at \gmat-win-R2022a\GMAT\data\vehicle\spad, added to repo
- SPAD files not usable for drag in GMAT according to [this link](https://docs.google.com/document/d/1tL2fp7NzYW6DZW6qtb0pLZYqV5YRxLy8AskRScsVRoM/edit?tab=t.0), to roll own drag model in Python
- Found that Python interface for GMAT requires 32-bit Python, to take note when creating venv
- First-cut model for basic orbit achieved
- Acquainted with thruster model parameters
- Learned Finite Burn modelling

# 2025-03-27
- Added basic thruster model
- Added basic finite burn model
- Added propellant scoop as negative-Isp thruster
- Added AIAA paper template
- Investigated using spacecraft properties in equations / scripts