This repository provides the experimental pipeline for poisoning-resilient stress detection using wearable physiological signals.

It implements:
- FG-DualNet (feature-gated neural architecture)
- Training-time poisoning attacks (label flipping, Gaussian noise, GAN-based injection)
- Defense mechanisms (Isolation Forest, CTGAN augmentation, SMOTE)
- Evaluation on DEAP (and extendable to WESAD)

The code supports reproducible experiments with detailed logging, performance metrics, and visualization outputs, aligned with the methodology presented in our paper on trustworthy AI for consumer wearable systems.
