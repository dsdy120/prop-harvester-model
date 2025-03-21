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
- 