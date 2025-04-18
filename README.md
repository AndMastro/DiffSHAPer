![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# DiffSHAPer

This is the official repository for the work **Explaining molecular diffusion models**.

We propose an explainability strategy based on Shapley values for diffusion models with linker desgin as a use case.

## Prerequisites

The code was developed and testes on an Ubuntu 22.04 system with Python 3.10.5 and PyTorch 2.4.0 (CUDA version 12.4). We believe other versions and systems to be compatible as well (let us know if you try other version as we will add them to the compatibility list!).

All the packages needed to run the code are available in ```environment.yml```. We suggest to use [Anaconda](https://www.anaconda.com/download) and create a new environment as

```bash
conda env create -f environment.yml
```

The enrivornment file provided within the repo was generated under Ubuntu. It may fail on Windows systems, so we recommend installing the needed packages separately in this case. If you have problems installing PyTorch from the environment, install following instructions [here](https://pytorch.org/get-started/locally/).

## Generate explanations

The repository contains the script ```diffusion_explainer.py```. It will generate linkers using [DiffLinker](https://github.com/igashov/DiffLinker) and explain the generated molecules using our approach.

The script will load parameters from ```config.yml```, where you can adjust several parameters as the number of samples to explain, the generation frames for which images should be generated, Monte Carlo sampling steps for Shapley values approximation, and others.

We also provide the notebook ```diffusion_explainer.ipynb``` for interactive sessions.

## Inspect results

Explanations are mapped to compounds structures, both in terms of graph representaitons and molecular graphs. The graph representation is generated inspired by visualization facilities from DiffLinker, while molecular graphs are generated using the [RDKit](https://www.rdkit.org/) library. An exemplary explanation with atom importance mapped onto molecular strucutres may look like this:

<p align="center">
  <img src="results/explanations/zinc_final_test/explanations_seed_42/mapping/structures/8/8_0_structure.png" alt="exemplary explanation mappedn onto molecular strucutres" width=45%>
</p>

We can see fragment atoms that favor (red) the generation of the linekr (green) and atoms that oppose it (blue). Mappings and computed Shapley values will be saved in the ```results/explanations/zinc_final_test/DATASET_NAME``` folder.

For any clarification, doubt, or curiosity, feel free to drop an [email](mailto:mastropietro@bit.uni-bonn.de)
