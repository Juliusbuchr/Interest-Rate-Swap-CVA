# CVA Calculation for Interest Rate Swaps Using the Hull-White Model

## Overview

This repository contains a Python script to compute the **unilateral Credit Valuation Adjustment (CVA)** for a 10-year, uncollateralized, plain vanilla interest rate swap (IRS) with quarterly payments, using the Hull-White model. It includes yield curve calibration, short rate simulation, exposure profile generation, default probability interpolation from Credit Default Swap (CDS) spreads, and CVA computation.

## Features

- **Hull-White Model Simulation**: Short rate simulation under the Hull-White model.
- **Yield Curve Calibration**: Curve fitting based on CIBOR, FRA, and swap market rates using Hermite interpolation.
- **Exposure Profiles**: Computes Expected Future Value (EFV), Expected Exposure (EE), Expected Negative Exposure (ENE), and Potential Future Exposure (PFE).
- **CVA Computation**: Uses default probabilities inferred from CDS spreads to calculate unilateral CVA for the swap.

## Requirements

Ensure you have the following dependencies installed:

- **FIDLIB**: A custom Fixed Income Derivatives library.
- Python libraries: `numpy`, `pandas`, `matplotlib`, `scipy`.

To install the required Python libraries, you can use:
```bash
pip install numpy pandas matplotlib scipy
