# kEDMD_Sig

This repository contains Python code for Koopman operator analysis using Extended Dynamic Mode Decomposition (EDMD) combined with signature methods.  
The method is applied to sea surface temperature (SST) data to extract decadal-scale modes, including dynamics related to the Atlantic Meridional Overturning Circulation (AMOC).

## Features
- Extended Dynamic Mode Decomposition (EDMD) with kernel methods
- Signature-kernel-based feature transformation for time series
- Koopman mode extraction from SST datasets
- Visualization tools for real and imaginary parts of the modes

## Directory Structure
kEDMD_Sig/
├── kernel_EDMD.py          # Kernel EDMD implementation
├── calc_gram-matrix.py     # Gram matrix calculation
├── make_sst-path.py        # Preprocessing of SST time series
├── data/                   # Input SST datasets (not included)
├── frames/                 # Output figures or animations
└── README.md

## Data Source

The SST dataset used in this project (`sst.mon.mean.nc`) is publicly available from NOAA PSL.

- **COBE-SST2 (JMA COBE2 SST dataset)**  
  [NOAA PSL COBE-SST2 page](https://psl.noaa.gov/data/gridded/data.cobe2.html)

### Downloading the dataset

You can download the dataset directly using the following `wget` command:

```bash
wget https://downloads.psl.noaa.gov/Datasets/COBE2/sst.mon.mean.nc -O data/sst.mon.mean.nc

## How to Use

1. Preprocess the SST time series:
    ```bash
    python make_sst-path.py
    ```

2. Compute the Gram matrix for kernel EDMD:
    ```bash
    python calc_gram-matrix.py
    ```

3. Run kernel EDMD analysis:
    ```bash
    python kernel_EDMD.py
    ```

Output visualizations (figures or animations) will be saved in the `frames/` directory.

## License
MIT License

