# Exoplanet Transit Detector
Python tool for detecting exoplanet transits in stellar light curve data and estimating planetary parameters.

## Overview

This project analyzes stellar light curves to identify periodic dips in brightness caused by transiting exoplanets. The program applies transit detection techniques to estimate key planetary parameters such as orbital period, planetary radius, and equilibrium temperature.

The code is designed to work with photometric light curve data from space missions such as the Kepler Space Telescope or TESS, but can be applied to any stellar brightness time series.

## Features
- Transit detection
- Orbital period estimation
- Planet radius estimation
- Orbital distance estimation
- Orbital Velocity estimation
- Planet equilibrium temperature estimation
- etc. 

## Usage
Run the program with:

```
python exoplanet.py
```

### Example Output

<p align="center"> <img src="Kepler-10_folded_lightcurve.png" width="600"> </p>

```
Detected period: 0.8375375375375377 d
Transit depth: 0.0001794067578906322
Estimated planet radius (Earth radii): 1.5563015764514885
Estimated orbital distance (AU): 0.01685086027862866
Estimated orbital velocity (km/s): 218.88164035175822
Estimated stellar luminosity (L/Lsun): 1.0244755227171471
Estimated equilibrium temperature (K): 1973.097816471603
Estimated equilibrium temperature (C): 1699.947816471603
Estimated stellar flux relative to Earth: 3607.9244205859054

```
