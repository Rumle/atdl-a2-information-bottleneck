# Reproducibility Study: Opening the Black Box of Deep Neural Networks via Information

This repository contains our group project investigating the reproducibility of **“Opening the Black Box of Deep Neural Networks via Information”** by Ravid Shwartz-Ziv and Naftali Tishby (2017).

## Project Objective

Following the NeurIPS Reproducibility Challenge guidelines, we aim to replicate the paper’s experiments and assess whether our results support its conclusions. We will document both successfully reproduced findings and discrepancies, together with possible explanations.

## Paper Overview

The paper studies neural network training through the **information plane**, which tracks the mutual information between hidden representations and the inputs and labels.

The authors propose that training involves two phases:

1. **Fitting:** The network learns information relevant to predicting the labels.
2. **Compression:** Hidden representations lose information about the input while retaining information relevant to prediction.

Our study examines whether selected experiments reproduce this reported behavior.

## Planned Experiments
- Information-plane dynamics: Train one fixed network and plot each layer’s I(X;T) and I(T;Y) over training.
- Gradient transition: Independently train a fixed network and track gradient mean and variability against epochs and training loss. This experiment uses its own runs, not the logs from Experiment 1 
- Depth and training time: Train a predeclared set of network depths and compare the time or number of updates needed to reach a fixed test-performance threshold. Choose the threshold in advance, without relying on results from the other experiments.
- Training-set size: Train one fixed architecture on predeclared sample fractions and compare test performance and, if feasible, final information-plane coordinates. This does not depend on the depth experiment.

The final experimental scope and any deviations from the original setup will be documented.

## Replication Approach

We follow the **Replication Track**, using the authors’ released code where available or reimplementing the selected methods from the paper. We will document the dataset, architecture, training settings, and mutual-information estimation procedure.

Targeted ablation studies may examine how factors such as information-estimation settings, training-set size, or network depth affect the observed behavior.

## Project Status

Work in progress. Experimental results and reproducibility conclusions will be added as the study develops.

## References

- Shwartz-Ziv, R., & Tishby, N. (2017). *Opening the Black Box of Deep Neural Networks via Information*. arXiv:1703.00810.
- [NeurIPS Reproducibility Challenge guidelines](https://reproml.org/neurips2019/).