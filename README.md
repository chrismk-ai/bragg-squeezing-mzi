# Code Repository: Balancing Quasi-Bragg Regime and Velocity Selectivity in Quantum-Enhanced Atom Interferometry

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21786457-blue.svg)](https://doi.org/10.5281/zenodo.21786457)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the numerical simulation code to generate the data presented in the paper:

> **Balancing Quasi-Bragg Regime and Velocity Selectivity in Quantum-Enhanced Atom Interferometry**  
> Published in *Physical Review Research*: [DOI: 10.1103/wr69-9hr5](https://doi.org/10.1103/wr69-9hr5)

---

## Repository Overview

* **`quantum_enhanced_Bragg_MZI.py`**  
  Contains all the core methods required to generate the data shown in the [paper](https://doi.org/10.1103/wr69-9hr5). An exemplary execution of these methods is provided in the `if __name__ == "__main__":` block at the end of the file.
  
  > **Note on Parallelization:** Running the script via the `if __name__ == "__main__":` block is strictly required because several simulation methods utilize parallel processing (multiprocessing).

* **`requirements.txt`**  
  Lists all necessary Python packages and their respective versions required to run `quantum_enhanced_Bragg_MZI.py`.

---

## Quick Start

### 1. Installation
Clone the repository and install the dependencies listed in `requirements.txt` optionally to a virtual environment:

```bash
# clone repo
git clone [https://github.com/chrismk-ai/bragg-squeezing-mzi.git](https://github.com/chrismk-ai/bragg-squeezing-mzi.git)
cd bragg-squeezing-mzi

# generate venv
python3 -m venv venv
source venv/bin/activate

# install requrements
pip install -r requirements.txt
