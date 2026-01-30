# AFM Multichannel Autoencoder for Unsupervised Feature Learning

This repository contains an **unsupervised deep learning pipeline** based on convolutional autoencoders, developed to analyze **multichannel AFM data** and explore shared and hidden representations across correlated physical signals.

The project focuses on **representation learning**, not classification accuracy, and was built on real experimental AFM datasets. In particular of thin film perovskites and diverse metal oxide films. 
---

## Motivation

In many AFM and scanning probe experiments, multiple channels are acquired simultaneously (e.g. topography, near-field signals, phase, or other local physical responses).

While these channels are often analyzed separately, they may contain **shared or complementary information** that is not obvious from direct inspection.

This project explores whether **unsupervised deep learning** can:

- learn compact latent representations of AFM images
- capture correlations across different physical channels
- reveal structure not visible in any single image modality

---

## Approach

A **convolutional autoencoder** is trained to reconstruct multichannel AFM images while compressing them into a low-dimensional latent space.

Key design choices:

- Unsupervised learning (no labels required)
- Multichannel inputs (each AFM signal as a separate channel)
- Convolutional architecture to preserve spatial structure
- Focus on latent-space analysis rather than reconstruction loss alone

---


## Project Structure

```text
afm-autoencoder/
├── src/                  # Model definition and training code
├── notebooks/            # Experiments and latent-space analysis
├── data/sample/          # Small illustrative dataset
├── outputs/              # Reconstructions and visualizations
├── requirements.txt
└── README.md
