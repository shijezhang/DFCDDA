# DFCDDA

DFCDDA is a diffusion-enhanced fine-grained cross semantic fusion framework for drug-disease association prediction.

Paper: [Diffusion-enhanced Fine-grained Cross Semantic Fusion for Drug-disease Association Prediction](https://ieeexplore.ieee.org/document/11554602)

## Framework

![DFCDDA framework](fig/framework%20diagram.png)

## Repository Structure

```text
DFCDDA_release/
├── main.py                         # Main cross-validation training entry
├── parse_args.py                   # Command-line configuration
├── data_preprocessing.py           # Dataset loading, preprocessing, KNN graph construction
├── model.py                        # DFCDDA prediction model
├── layers.py                       # Heterogeneous graph and semantic fusion layers
├── contrastive_learning.py         # Intra- and inter-view contrastive learning losses
├── diffusion/
│   ├── gaussian_diffusion.py       # Gaussian diffusion process
│   ├── resample.py                 # Diffusion timestep samplers
│   ├── respace.py                  # Diffusion timestep respacing
│   ├── losses.py                   # Diffusion loss utilities
│   └── nn.py                       # Diffusion neural network helpers
├── util/
│   ├── evaluate.py                 # Evaluation metrics
│   ├── script_util.py              # Diffusion configuration helpers
│   └── unet.py                     # Conditional U-Net backbone
├── dataset/
│   ├── B-dataset/                  # Benchmark dataset
│   ├── C-dataset/                  # Benchmark dataset
│   └── F-dataset/                  # Benchmark dataset
└── fig/
    └── framework diagram.png       # Framework diagram used in this README
```

## Installation

Create a clean Python environment and install the required dependencies:

```bash
conda create -n dfcdda python=3.8.1
conda activate dfcdda
conda install pytorch==1.9.1 cudatoolkit=11.3 -c pytorch
pip install numpy==1.24.4 pandas==2.0.3 scikit-learn==1.3.2
```

DGL wheels are tied to the local PyTorch and CUDA setup. If the command above does not match your CUDA environment, install the DGL build that matches your PyTorch and CUDA versions before running the model.

## Quick Start

Run the default 10-fold experiment on the C-dataset:

```bash
python main.py --dataset C-dataset --device 0
```

Run another benchmark dataset:

```bash
python main.py --dataset B-dataset --device 0
python main.py --dataset F-dataset --device 0
```

Training prints fold-level AUROC, AUPR, accuracy, precision, recall, F1, and MCC during optimization.

## Dataset

Each benchmark folder under `dataset/` contains the association networks, similarity matrices, and pretrained features used by DFCDDA:

| File | Description |
| --- | --- |
| `DrugDiseaseAssociationNumber.csv` | Validated drug-disease associations |
| `DrugProteinAssociationNumber.csv` | Validated drug-protein associations |
| `ProteinDiseaseAssociationNumber.csv` | Validated protein-disease associations |
| `DrugFingerprint.csv`, `DrugGIP.csv` | Drug-drug similarity matrices |
| `DiseasePS.csv`, `DiseaseGIP.csv` | Disease-disease similarity matrices |
| `Drug_mol2vec.csv` | Drug feature embeddings |
| `DiseaseFeature.csv` | Disease feature embeddings |
| `Protein_ESM.csv` | Protein feature embeddings |
| `AllNode.csv`, `Alledge.csv`, `adj.csv` | Auxiliary graph files for benchmark construction |

## Key Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--dataset` | Benchmark dataset: `F-dataset`, `C-dataset`, or `B-dataset` | `C-dataset` |
| `--device` | CUDA device index | `0` |
| `--K_fold` | Number of cross-validation folds | `10` |
| `--lr` | Learning rate | `1e-3` |
| `--weight_decay` | Adam weight decay | `1e-4` |
| `--knn_neighbor` | KNN size for drug and disease similarity graphs | `5` |
| `--negative_rate` | Negative-to-positive sampling ratio | `1` |
| `--alpha` | Diffusion loss weight | `1` |
| `--beta` | Contrastive loss weight | `1e-5` |

Run `python main.py --help` for the full configuration, including diffusion and model architecture options.

## Reproducibility Notes

The training script fixes NumPy and PyTorch random seeds through `--seed`. Reported performance may still vary slightly across hardware, CUDA kernels, PyTorch/DGL versions, and negative sampling settings. For manuscript-scale reproduction, use the default 10-fold setting and the benchmark datasets provided in this release.

## Citation

If this code supports your research, please cite the corresponding DFCDDA paper:

[Diffusion-enhanced Fine-grained Cross Semantic Fusion for Drug-disease Association Prediction](https://ieeexplore.ieee.org/document/11554602)
