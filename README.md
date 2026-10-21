# Brain connectivity classification with graph neural networks

Research code for classifying functional brain connectivity graphs from PPMI and ADNI fMRI data. The models combine connectivity with brain context: functional groups, hemisphere information, and optional asymmetry features.

The implementation includes GIN, GCN, GAT, GPS, and MLP variants in PyTorch and PyTorch Geometric.

## Research context

- [Functional-Positional Context Embedding in Graph Neural Networks for Improved Parkinson's Disease Diagnosis](https://doi.org/10.1109/ISBI60581.2025.10980965), IEEE ISBI 2025.
- [Edge-Boosted Graph Learning for Functional Brain Connectivity Analysis](https://arxiv.org/abs/2504.14796), IEEE ISBI 2025.

This repository provides the brain connectivity framework. The papers describe their specific experiments; the repository does not include the source datasets, trained checkpoints, or an archived run reproducing every published result.

## How the code is organized

| File | Responsibility |
| :--- | :--- |
| [dataset.py](dataset.py) | Read preprocessed subject time series and construct correlation graphs |
| [preprocessing.py](preprocessing.py) | Threshold connectivity and construct graph attributes |
| [networks.py](networks.py) | Define functional network groups |
| [models.py](models.py) | Brain context layer and model variants |
| [train.py](train.py) | Cross-validation, training, evaluation, and logging |
| [main.py](main.py) | Command-line experiment settings |

## Setup

```bash
git clone https://github.com/MostafaAbdelmegeed/brain_connectome.git
cd brain_connectome
conda env create -f environment.yml
conda activate clean_brain
```

The environment file lists the dependencies; it is not a version-locked snapshot of the paper experiments. Select a PyTorch build compatible with your machine if Conda needs a different CPU/CUDA configuration.

## Input data

The loaders expect **preprocessed region time series**, rather than raw MRI scans. Obtain access to the datasets through [PPMI](https://www.ppmi-info.org/access-data-specimens/download-data) or [ADNI](https://adni.loni.usc.edu/data-samples/adni-data/) and follow their data-use requirements.

```text
data/
  PPMI/
    control/
      subject_AAL116_features_timeseries.mat
    patient/
      subject_AAL116_features_timeseries.mat
    prodromal/
      ...
    swedd/
      ...
  ADNI/
    AAL90/
      sub_<subject_id>_...
    label-2cls_new.csv
```

PPMI MATLAB files need a `data` array with time points by regions. Their parent directory names determine the labels: `control`, `patient`, `prodromal`, and `swedd`. ADNI time-series files are read with `numpy.loadtxt`; the CSV needs `subject_id` and `DX` columns. See the loaders for the class mappings.

## Run an experiment

Run commands from the repository root. For example:

```bash
python main.py --dataset ppmi --model gin --seed 0 --n_folds 10 --epochs 300 --batch_size 64 --learning_rate 0.0001
```

For functional and hemisphere context, select the relevant mode and network:

```bash
python main.py --dataset ppmi --model gin --mode both --network yeo7 --include_asymmetry
```

ADNI supports two- or four-class mappings. Select the class count explicitly:

```bash
python main.py --dataset adni --model gin --num_classes 2
```

Use `python main.py --help` for the remaining settings, including graph variants, edge attributes, sparsity, dropout, and attention heads. Models and context options differ; check [get_model](models.py) when combining flags.

## Evaluation

The training loop uses stratified folds and records accuracy, precision, F1, and confusion matrices. Compare models with the same subject inputs, class mapping, split settings, and seed. Inspect the logged runs before drawing conclusions from a single score.

```bash
tensorboard --logdir polished/new_runs/
```

For questions about the implementation: [mostafa0as3d@gmail.com](mailto:mostafa0as3d@gmail.com).
