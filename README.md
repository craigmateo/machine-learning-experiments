# Machine Learning Experiments

This repository contains a collection of applied machine learning experiments exploring
supervised and unsupervised models across multiple datasets.

The focus is on **model behavior, assumptions, and evaluation**, rather than on reproducing
tutorials or completing platform-specific coursework.

---

## Purpose

The goal of this repository is to:

- explore how different machine learning models behave under varying assumptions
- compare algorithms across datasets and feature representations
- understand trade-offs between model complexity, performance, and interpretability
- document empirical observations and modeling decisions

Some experiments are inspired by structured courses and tutorials, but all implementations
are adapted, extended, or recontextualized to support independent exploration.

---

## Scope

Experiments in this repository may include:

- **Supervised learning**
  - regression
  - classification
  - model evaluation and error analysis

- **Unsupervised learning**
  - clustering
  - dimensionality reduction
  - distance metrics and similarity assumptions

- **Model evaluation**
  - cross-validation
  - bias / variance behavior
  - metric selection and interpretation

---

## Repository Structure

machine-learning-experiments/
├── datasets/ # Raw and processed datasets (where permitted)

├── notebooks/ # Exploratory analyses and model experiments

│ ├── regression/

│ ├── classification/

│ ├── clustering/

│ └── dimensionality-reduction/

├── src/ # Reusable preprocessing, modeling, and evaluation code

├── experiments.md # Log of experiments, decisions, and observations

└── README.md


---

## Experiment Log

A running log of experiments is maintained in `experiments.md`, documenting:

- the question or hypothesis being explored
- the dataset and features used
- the models and parameters tested
- observed outcomes and surprises
- follow-up questions or next steps

This log is intended as a **research notebook**, not a polished report.

---

## Notes on Reproducibility

- Datasets are referenced where possible; large or restricted datasets may be excluded
- Random seeds are fixed when appropriate
- Results are interpreted qualitatively unless otherwise stated

---

## Status

This repository is **actively evolving**.  
Experiments may be partial, exploratory, or superseded over time.
