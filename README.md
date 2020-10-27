# Hybrid Monte Carlo

This repository includes Pyhon modules and Jupyter notebooks for Monte Carlo simulation of financial instruments.

We provide a set of models that can be combined to form hybrid models for interest rates, FX and equities. Financial instruments are represented by individual payoff objects. The payoff objects are combined to form the contractual cash flows of the instrument.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Recommended Python environment is Anaconda. This project is based on Python and Jupyter Notebook. Important packages are Numpy, Pandas and Plotly and QuantLib.

### Installing

Clone the repository via:

```
git clone https://github.com/sschlenkrich/HybridMonteCarlo.git
```

Start Anaconda, open a terminal and navigate to the folder of the local repository.

```
cd [path-to]/HybridMonteCarlo/
```

Create a new Anaconda Python Environment with required packages 

```
conda create --name [envname] --file Requirements.txt
```

Activate the new environment via

```
conda activate [envname]
```

Install QuantLib library via pip (QuantLib is not available via conda)

```
pip install quantlib
```

Now start Jupyter Notebook with

```
jupyter notebook
```

This step should open a browser window. Navigate to

```
[path-to]/HybridMonteCarlo/doc/MonteCarloSimulation.ipynb
```

Finally, the notebook can be executed by selecting the menu item

```
Kernel > Restart & Run all
```

Inspect the output and have fun.

## Repository Structure

  -  **doc/** - documentation and example jupyter notebooks
  -  **hybmc/** - source code
       -  **mathutils/** - auxilliary methods for implied volatility and linear regression
       -  **models/** - financial models used for simulation
       -  **products/** - financial products that provide payoff structures
       -  **simulations/** - Monte Carlo simulation and payoffs
       -  **termstructures/** - interest rate and volatility structures

## Data Files

There are currently no data files included in the project.

Models can be serialised and saved via pickle. We use this approach in the notebooks.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/sschlenkrich/HybridMonteCarlo/tags). 

## Authors and History

* **Sebastian Schlenkrich**, October 2020 - *Initial setup*
