# MSc Thesis: Safety Stock Optimization via Deep Reinforcement Learning

This repository contains the code and resources for the MSc thesis titled **"Safety Stock Optimization via Deep Reinforcement Learning"**.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Repository Structure

The repository is organized as follows:

### Final/

This folder contains the final implementation of the models, training processes, and evaluations, along with any generated data, trained models, and log files.
- **Archive/**: Contains code that was either incomplete or not used in the final implementation.
- **Config/**: Configuration files for the environment and network settings.
- **Data/**: Generated simulation data used for visualization.
- **Environment/**: Code for the environment.
- **Version 9 (V9)**: Final version used in the thesis.
- **Version 10 (V10)**: Extended version capable of handling multi-level supply chains.
- **Functions/**: Contains algorithms for evaluation or visualization. Includes discarded implementations and utility functions.
- **Training/**: Saved training data and models, as well as results from the optimization process.

### Notes/

Contains notes and scripts used during model development and technology familiarization.

### Prework/

Initial draft of a potential environment, developed in collaboration with MSD.

### Testing/

Code and scripts for testing the models and environment before moving to the final implementation phase.

## Jupyter Notebooks

The repository includes several Jupyter notebooks for different stages of model development and evaluation:
- **00_evaluation.ipynb**: Evaluation of models and benchmarking against other policies.
- **00_safety_stock_optimization.ipynb**: Analysis of safety stock results for comparison purposes.
- **01_*.ipynb**: Training notebooks for the models. Prepared for execution on Kaggle Notebooks.
- **02_*.ipynb**: Evaluation and visualization of trained models.
- **03_*.ipynb**: Implementation of benchmark policies (e.g., greedy algorithm, fixed-policy, calculated safety stock).
- **04_*.ipynb**: Plots and results from hyperparameter optimization.
- **05_*.ipynb**: Code for optimization and sensitivity analysis of the environment parameters.
- **11_*.ipynb & 12_*.ipynb**: Code for multi-level supply chain environments and their training/evaluation using PPO.
- **99_*.ipynb**: Notebooks used for testing and environment tuning.

## Final Implementation Details

The final implementation (Final/) includes:
- **Environment Code**: Versions 9 (used in the thesis) and 10 (extended multi-level supply chain).
- **Simulation Data**: Used for visualization and evaluation.
- **Saved Models and Results**: From training and optimization processes.