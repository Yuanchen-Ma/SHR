# SHR: Selective Regression with Historical Risk

This package accompanies *Selective remaining useful life prediction using conditional moments and historical information*, prepared for *Reliability Engineering & System Safety*. It implements SHR and five comparison methods for TCN and Transformer predictors on C-MAPSS and N-CMAPSS.

## Environment

From the `SHR` directory, create a Conda environment with Python 3.12.

```text
conda create -n shr python=3.12 pip
conda activate shr
```

For GPU training with CUDA 13.0 on Windows or Linux, install PyTorch before installing the project. Other CUDA builds are listed in the PyTorch instructions. [PyTorch 2.11.0 installation](https://pytorch.org/get-started/previous-versions/#v2110)

```text
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

Install the project and its pinned dependencies from the checkout.

```text
python -m pip install -e .
```

## Data

Download entries 6 and 17. If you use the default paths, create the directories below when unpacking the data and keep the original filenames. If you already have the data elsewhere, use the path options in the run instructions. [NASA Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

- **C-MAPSS.** Put `train_FDxxx.txt`, `test_FDxxx.txt`, and `RUL_FDxxx.txt` for each of FD001, FD002, FD003, and FD004 under `data/raw/CMAPSSData/`. The corresponding FD001–FD004 subdirectories also work. [Download ZIP](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) · [Benchmark paper](https://doi.org/10.1109/PHM.2008.4711414)
- **N-CMAPSS.** Put `N-CMAPSS_DS01-005.h5`, `N-CMAPSS_DS02-006.h5`, `N-CMAPSS_DS03-012.h5`, `N-CMAPSS_DS04.h5`, `N-CMAPSS_DS05.h5`, `N-CMAPSS_DS06.h5`, and `N-CMAPSS_DS07.h5` under `data/raw/NCMAPSSData/`. The seven files occupy about 21 GB. [Download ZIP](https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip) · [Data descriptor](https://doi.org/10.3390/data6010005)

## Run

```text
python -m shr plan
python -m shr upstream --stage check-data
python -m shr reproduce --device cuda --workers 1
```

The first two commands show the experiment plan and validate the input files. The last trains the models and evaluates the selectors. The workflow creates `experiments/` and `results/` as needed. Pass `--cmapss-dir PATH` and `--ncmapss-dir PATH` to use existing data directories. Use `--device cpu` if CUDA is unavailable. To resume an interrupted run, rerun the same command with the same data, device, environment, and settings.
