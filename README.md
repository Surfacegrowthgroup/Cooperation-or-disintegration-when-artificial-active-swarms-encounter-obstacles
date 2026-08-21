# Numerical program codes for “Cooperation or disintegration when artificial active swarms encounter obstacles?”

## Contents

`src` is the current implementation of the numerical experiments in this work. It solves a discrete-time Vicsek model with static quenched obstacles, using open boundaries in the longitudinal `x` direction and periodic boundaries in the transverse `y` direction. The code separates each experiment into independent stages for solving, batch scheduling, source-data storage, postprocessing, and visualization:

- `settings.py` stores all physical and temporal parameters, while `config.py` defines scan, scheduling, and postprocessing configurations.
- `simulation.py` and `vm_engine.py` provide the serial solver and Numba-accelerated neighbour calculations.
- `controller.py` manages independent repetitions, one-parameter scans, and multiprocessing.
- `storage.py` writes complete trajectories and reproducibility metadata as HDF5 v1 source files.
- `postprocess.py` computes the passage rate, order parameters, and clustering coefficients from source trajectories.
- `animate.py`, `feature_plot.py`, and `cli.py` provide morphology animation, feature plotting, and a unified command-line interface, respectively.

By default, source trajectories are written to `data/raw/<submission>/`, and feature files are written to `data/processed/<submission>/`. Both are NAS-backed persistent project content. Temporary figures are written by default to `tmp/feature-plots/<submission>/`.

### Running environment

To reproduce the numerical environment, create the Conda environment specified by `environment.yml`:

```powershell
conda env create -f environment/environment.yml
conda activate cooperation-reproduction
```

The environment is based on Python 3.9.12 and pins `numpy=1.22.4`, `matplotlib=3.5.1`, `scipy=1.13.1`, `networkx=2.7.1`, and `tqdm=4.67.0`. The current module also requires `h5py` for HDF5 source data, `numba` for neighbour calculations, `psutil` for parallel progress reporting, and `scienceplots` for figure styling. `ffmpeg` is required when exporting MP4 animations; all of these dependencies are listed in the environment file.

This directory follows a `src` layout. If the command-line entry point has not been installed, add `src` to the Python path from the project root and invoke the module directly:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m encounter.cli run --help
```

If the `cooperation` command has already been registered in your environment, it can be used as an equivalent replacement for `python -m encounter.cli` in the commands below.

### Settings for numerical programs

All solver parameters are controlled by `EncounterSettings` in `settings.py`. Override defaults from the command line with repeatable `--set name=value` arguments, for example `--set times=6000 --set length=400`. Their meanings and default values are listed below.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `par_density` | 0.1 | Density of active particles, $\rho$ |
| `que_density` | 0.01 | Density of quenched obstacles, $\rho_{\mathrm{o}}$ |
| `times` | 6000 | Total number of time steps in one simulation, $T$ |
| `width` | 50 | Transverse system width, $W$ |
| `length` | 400 | Longitudinal length of the obstacle region, $L_{\mathrm{o}}$ |
| `left_length` | -100 | Longitudinal extent of the left display region |
| `place_length` | 50 | Longitudinal length of the initial particle-placement region, $L_{\mathrm{p}}$ |
| `white_length` | 10 | Buffer-region length between the placement and obstacle regions, $L_{\mathrm{w}}$ |
| `end_length` | 100 | Longitudinal length of the right-end display region, $L_{\mathrm{e}}$ |
| `radius` | 1 | Active-particle interaction radius, $R_{\mathrm{a}}$ |
| `strength` | 0.01 | Amplitude of annealed thermal noise, $\eta$ |
| `speed` | 0.3 | Particle displacement per time step, $v_0$ |
| `que_stren` | 1 | Amplitude of quenched noise, $H$ |
| `que_radius` | 0.5 | Quenched-obstacle interaction radius, $R_{\mathrm{o}}$ |

The particle number is derived as $N=\lfloor W L_{\mathrm{p}}\rho\rfloor$, and the obstacle number as $N_{\mathrm{o}}=\lfloor W L_{\mathrm{o}}\rho_{\mathrm{o}}\rfloor$. The obstacle region begins at `place_length + white_length`. These derived quantities cannot be overridden directly with `--set`. At each step, particle orientations are updated before positions are advanced by `speed`; `y` positions wrap periodically, and angles are normalized to $[-\pi,\pi]$.

### Drawing of morphology

Morphology visualization reads an existing HDF5 source file only: it neither solves the model again nor modifies the source data. `animate` exports a GIF or MP4 animation, and `snapshot` exports a single PDF, PNG, or SVG frame at a specified physical time step:

```powershell
python -m encounter.cli animate <source.h5> --frame-step 20 --fps 5 --output evolution.gif
python -m encounter.cli snapshot <source.h5> --step 3000 --output snapshot.pdf
```

The defaults are `--frame-step 20`, `--fps 5`, and `--dpi 160`. `--color-mode direction` uses a cyclic colormap to encode motion direction, whereas the default `uniform` uses a single particle colour. `--aspect-mode equal` preserves the exact physical aspect ratio, while the default `readable` improves the visibility of elongated systems. If `snapshot --step` is omitted, the last frame is rendered; use `--label "(a)"` to add a panel label.

### Simulation and data computation

A fixed-parameter run produces one HDF5 source trajectory. With `--scan parameter:start:stop:step`, the program performs an equally spaced, endpoint-inclusive scan over one numerical parameter. `--loop-times` specifies the number of independent repetitions per scan point, and `--workers` specifies the number of parallel worker processes. When `--seed` is supplied, every repetition–scan-point pair receives a deterministic child seed, so the results do not depend on the worker count or scheduling order.

```powershell
# One submission with default physical parameters; `run` may be omitted.
python -m encounter.cli --set times=6000 --set length=400 --name default-run --seed 42

# Scan the obstacle-region length: 25 scan points, each repeated 10 times.
python -m encounter.cli run --name length-scan --scan length:0:1200:50 --loop-times 10 --workers 4 --seed 42
```

Each scan-point–repetition pair produces one HDF5 file. Before execution, all target paths are prechecked. Existing output is never overwritten; use a new `--name` or output directory instead.

Postprocessing computes the final passage rate $R$, together with the order parameters $V_{1,2}$ and clustering coefficients $C_{1,2}$ on the left (returning) and right (passing) sides of the obstacle region. Apart from $R$, which is evaluated at the final time, scalar quantities are averaged over steps from `--start-average` onward; the default starting step is 3000. Processing an individual HDF5 file creates a matching `.features.npz`; processing a complete submission directory validates the scan–repetition matrix, creates any missing per-run feature files, and writes `summary.npz`.

```powershell
# Either an individual run or a complete submission directory may be used as input.
python -m encounter.cli postprocess data/raw/encounter/length-scan
python -m encounter.cli postprocess <source.h5> --start-average 3000

# Plot feature files created during postprocessing.
python -m encounter.cli plot data/processed/encounter/<submission>/<run>.features.npz
python -m encounter.cli plot data/processed/encounter/<submission>/summary.npz --output-dir tmp/feature-plots
```

A per-run feature file produces two time-series figures, for returning and passing particles. A scan summary produces three parameter-scan figures: passage rate, returning particles, and passing particles. The plotting module reads NPZ feature files only; it neither reloads HDF5 trajectories nor recomputes features. It also refuses to overwrite an existing target PDF.
