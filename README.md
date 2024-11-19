# Non-Linear Outlier Synthesis

Code used for our paper [Non-Linear Outlier Synthesis for Out-of-Distribution Detection](https://arxiv.org/pdf/2407.04022).

![Example Outliers](figures/teaser.png "Example Outliers")

This repository contains the code to obtain diffusion embeddings, train our conditional volume-preserving networks, sample outlier embeddings, synthesize the outliers and train and test classifiers.

### Abstract

The reliability of supervised classifiers is severely hampered by their limitations in dealing with unexpected inputs, leading to great interest in out-of-distribution (OOD) detection. Recently, OOD detectors trained on synthetic outliers, especially those generated by large diffusion models, have shown promising results in defining robust OOD decision boundaries. Building on this progress, we present NCIS, which enhances the quality of synthetic outliers by operating directly in the diffusion's model embedding space rather than combining disjoint models as in previous work and by modeling class-conditional manifolds with a conditional volume-preserving network for more expressive characterization of the training distribution. We demonstrate that these improvements yield new state-of-the-art OOD detection results on standard ImageNet100 and CIFAR100 benchmarks, and provide insights into the importance of data pre-processing and other key design choices.


### Comments

Our codebase builds on the [Dream-OOD](https://github.com/deeplearning-wisc/dream-ood/tree/main) and [Non-linear invariants](https://github.com/LarsDoorenbos/Nonlinear-invariants) repositories.

### Citation

If you find this work helpful, consider citing it using

```
@article{doorenbos2024learning,
  title={Learning non-linear invariants for unsupervised out-of-distribution detection},
  author={Doorenbos, Lars and Sznitman, Raphael and M{\'a}rquez-Neila, Pablo},
  journal={arXiv preprint arXiv:2407.04022},
  year={2024}
}
```
