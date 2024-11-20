# Brain Connectivity Graph Classification with GNNs

This repository provides a framework for classifying brain connectivity graphs derived from the PPMI and ADNI datasets. It utilizes Graph Neural Networks (GNNs) with a focus on incorporating domain-specific knowledge for analyzing functional connectivity matrices.

---

## Features

- **Dataset Handling**:
  - PPMI and ADNI datasets for neurodegenerative disease classification.
  - Functional connectivity graphs constructed from BOLD fMRI signals.
- **Graph Neural Networks**:

  - Models include GIN, GCN, GAT, GPS, and MLP variants.
  - Support for advanced features like edge attributes and functional embeddings.

- **Custom Brain Context Layer**:

  - Includes embeddings for functional groups, hemispheric information, and asymmetry features.

- **Cross-Validation**:
  - Built-in support for cross-validation and stratified data splits.
  - Logging and visualization using TensorBoard.

---

## Repository Structure

```
.
├── dataset.py          # Dataset loading and preprocessing for PPMI and ADNI
├── main.py             # Entry point for training and evaluation
├── models.py           # Implementation of GNN-based models
├── networks.py         # Functional group definitions (Yeo networks)
├── preprocessing.py    # Preprocessing utilities for adjacency matrices and edge features
├── train.py            # Training pipeline with metrics and TensorBoard integration
```

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch and PyTorch Geometric
- Required Python libraries (`numpy`, `scipy`, `pandas`, `sklearn`, etc.)

To set up the environment, use:

```bash
conda env create -f environment.yml
conda activate brain-gnn
```

### Installation

Clone this repository:

```bash
git clone https://github.com/your-repo-url/brain-connectivity-gnn.git
cd brain-connectivity-gnn
```

---

## Usage

### Data Preparation

Ensure your dataset is organized as follows:

- **PPMI Dataset**:
  ```
  data/
  └── PPMI/
      ├── sub_<id>_AAL116_features_timeseries.mat
      ├── ...
  ```
- **ADNI Dataset**:
  ```
  data/
  └── ADNI/
      ├── AAL90/
      ├── label-2cls_new.csv
      ├── ...
  ```

### Training

To train a GNN model on the PPMI dataset:

```bash
python main.py --dataset ppmi --model gin --seed 0 --n_folds 10 --epochs 300 --batch_size 64 --learning_rate 0.0001
```

Key arguments:

- `--model`: Model to use (`gin`, `gcn`, `gat`, `gps`, `mlp`).
- `--dataset`: Dataset to train on (`ppmi`, `adni`).
- `--include_asymmetry`: Include asymmetry features.
- `--use_edges`: Use edge attributes.
- `--hidden_dim`: Hidden layer dimensions.
- `--dropout`: Dropout rate.

Check `main.py` for all configurable arguments.

---

## Functional Networks

### Yeo 7 and 17 Networks

Functional groups derived from Yeo 7 and 17 parcellations:

- Visual, Somatomotor, Limbic, Default Mode, and more.
- Automatically computed and assigned to brain regions.

---

## Results and Metrics

The framework evaluates models with:

- **Accuracy**
- **Precision**
- **F1-Score**
- **Confusion Matrix Entropy**

TensorBoard is used for visualization:

```bash
tensorboard --logdir polished/new_runs/
```

---

## Examples

### Sample Training Command

```bash
python main.py --dataset adni --model gps --n_layers 3 --dropout 0.5 --heads 4 --epochs 200
```

### Sample TensorBoard Output

- View metrics like loss, accuracy, and F1-score for each fold.
- Inspect the confusion matrix and interpret model attention weights.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or collaborations:

- **Email**: your-email@example.com
- **GitHub**: [Your GitHub Profile](https://github.com/your-profile)
