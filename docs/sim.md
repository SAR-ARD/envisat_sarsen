# Setup and Development Guide

This guide provides instructions for setting up the asar-xarray project for development and use.

## Prerequisites

Before you begin, ensure you have the following installed:

* conda (Miniconda or Anaconda)

---

## Installation Steps

**1. Clone the repository:**

```bash
git clone [https://github.com/Achaad/envisat_sarsen](https://github.com/Achaad/envisat_sarsen.git)
cd envisat_sarsen
```

**2. Create and activate a conda environment:**

```bash
conda env create -f environment.yml
conda activate envisat_sarsen
```

**3. Install the package in editable mode:**

```bash
pip install -e .
```