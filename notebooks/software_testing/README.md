# Stress-Testing Software
 
Concerns have been raised about the robustness of FNO software implementation across different hardwares in the `neuraloperator` package V2.0.0 (avaliable at the time of experimentation). Therefore, we upgrade the `neuraloperator` package and rerun a subset of key experiments (found in `demo.py`).
 
An indepth discussion of zero-shot super-resolution vs. multi-resolution training on different hardware architecture is provided in [Appendix N](https://arxiv.org/pdf/2510.06646). TLDR: We observe a wide variance in error magnitudes across different hardware+software versions. Further, we observe the predicted spectra of the zero-shot model consistently diverged from the label spectra after the maximum observable frequency in the training data. The core trend of multi-resolution training outperforming zero-shot super resolution is consistent across computing architectures+software stacks. While implementation differences in hardware-software stacks give rise to different levels of error in the model’s forward pass it does so consistently across different types of model training regimes (e.g., single-resolution vs. multi-resolution).  

 - `demo_updated_neuraloperator_package.ipynb`: same expriments as `../demo.py`, run on NVIDIA 100 GPU w/ updated `neuraloperator` library (commit: 98cd305099f4a2b232ed85773984f3e5991f9b1a)

We thank Dr. Valentine Duruisseaux for bringing to our attention an updated version of the `neuraloperator` library (commit: 98cd305099f4a2b232ed85773984f3e5991f9b1a) which addresses issues w.r.t. enforcing Hermitian symmetry when taking the irfftn in the spectral convolution layers in FNO. 
