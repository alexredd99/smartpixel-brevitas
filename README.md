# Smart Pixel - Brevitas

This repo re-implements and trains edge NNs and ParT using quantization-aware training techinques (Brevitas). Built for Arbolta.

# Models

Edge NN Models used in this repo based on [edge-nns](https://github.com/KastnerRG/edge-nns.git). Instead of original [fkeras](https://github.com/KastnerRG/fkeras.git), these models are built based on PyTorch and quantisized using [Brevitas](https://github.com/Xilinx/brevitas.git).

Particle Transformer (ParT) is a transformer model used for jet tagging. ParT's structure is proposed in [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772). Our model is a quantized model.

# Dataset

Smart Pixel dataset can be downloaded from [here](https://cseweb.ucsd.edu/~oweng/smart_pixel_dataset/).

JetClass dataset can be downloaded from [here](https://zenodo.org/records/6619768).

# Environment

References include `./requirement.txt` and `./environment.yaml` (in progress).
