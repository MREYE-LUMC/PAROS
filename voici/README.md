# PAROS demo using Voici

This notebook is intended to be rendered using [voici](https://voici.readthedocs.io/en/latest/)
as a standalone web interface to PAROS.

## How to build

The web app is built automatically using GitHub Actions.
If you want to build it manually, follow the steps outlined below.
These steps are only tested on Linux (and likely to fail on Windows).

1. Install [Miniforge](https://github.com/conda-forge/miniforge#install):
   ```shell
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
2. Make sure you are in the `voici` directory:
   ```shell
   cd voici/
   ```
3. Create the build environment:
   ```shell
   mamba env create -n paros-voici-build -f environment.yml
   ```
4. Build the notebook:
   ```shell
   voici build --contents=simple_example.ipynb
   ```
5. Optionally, run a web server to test the web app:
   ```shell
   python -m http.server 8000 --directory _output
   ```
