import torch
import numpy as np
import datetime

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, confusion_matrix
import torch.nn as nn
from torch_geometric.nn import summary
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from dataset import Dataset_PPMI, PPMIAsymmetryDataset, ADNIAsymmetryDataset, Dataset_ADNI
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Subset

# from dataset import collate_function
from models import get_model

import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image

from preprocessing import process

def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise, use black
    threshold = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    # Save the plot to a PNG in memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    
    # Convert to a format that TensorBoard can handle
    image = np.array(image)
    
    # Ensure the image is in RGB format (height, width, 3)
    if len(image.shape) == 2:  # Grayscale to RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:  # RGBA to RGB
        image = image[..., :3]
    
    return image


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax probability of the true class

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            ce_loss = ce_loss * at

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(args, device):
    # Hyperparameters
    model_name = args.model
    n_folds = args.n_folds
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    patience = args.patience
    heads = args.heads
    n_layers = args.n_layers
    dataset = args.dataset
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    # Initialize lists to store metrics for all folds
    all_fold_metrics = {'accuracy': [], 'precision': [], 'f1': [], 'conf_matrix':[], 'val_loss': [], 'weights': []}

    if args.mode == 'corr' or args.mode == 'func':
        dataset = Dataset_PPMI('data/PPMI') if args.dataset == 'ppmi' else Dataset_ADNI('data/ADNI/AAL90', 'data/ADNI/label-2cls_new.csv', args.num_classes)
    elif args.mode == 'asym' or args.mode == 'all':
        dataset = PPMIAsymmetryDataset('data/PPMI') if args.dataset == 'ppmi' else ADNIAsymmetryDataset('data/ADNI/AAL90', 'data/ADNI/label-2cls_new.csv', args.num_classes)
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    # TensorBoard writer
    run_name = args.run_name if args.run_name else f'{model_name}_{args.exp_code}_{args.dataset}_s{seed}_f{n_folds}_e{epochs}_bs{batch_size}_lr{learning_rate}_hd{hidden_dim}_d{dropout}_h{heads}_l{n_layers}_a{args.augmented}_{timestamp}'
    writer = SummaryWriter(log_dir=f'polished/runs/{run_name}')
    writer.add_text('Arguments', str(args))

    generator = torch.Generator().manual_seed(seed)
    # Training and evaluation loop
    for fold, (train_index, test_index) in enumerate(skf.split(dataset.data, dataset.labels)):
        print_with_timestamp(f"Fold {fold + 1}/{n_folds}")
        train_data = Subset(dataset, train_index)
        val_data = Subset(dataset, test_index)
        # Extract labels from each dataset
        train_labels = [data.y.item() for data in train_data]
        val_labels = [data.y.item() for data in val_data]

        # Calculate label distributions
        train_label_distribution = np.bincount(train_labels)
        val_label_distribution = np.bincount(val_labels)

        # Print label distributions
        print_with_timestamp(f"Training labels distribution: {train_label_distribution}")
        print_with_timestamp(f"Validation labels distribution: {val_label_distribution}")

        train_loader = DataLoader(train_data, batch_size=batch_size, generator=generator, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, generator=generator)

        model = get_model(args).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=learning_rate)


        print_with_timestamp(f"Epoch/Loss\t||\tTraining\t|\tValidation\t")
        criterion = torch.nn.CrossEntropyLoss().to(device)

        best_val_loss = float('inf')
        best_val_acc = 0
        best_val_f1 = 0
        best_val_precision = 0
        patience_counter = 0
        best_confusion_matrix = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct_predictions = 0
            total_predictions = 0
            all_preds = []
            all_labels = []
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output, _ = model(data)
                loss = criterion(output, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                preds = output.argmax(dim=1)
                correct_predictions += (preds == data.y).sum().item()
                total_predictions += data.y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            
        

            train_loss /= len(train_loader)
            writer.add_scalar(f'Fold_{fold+1}/Metrics/Train_Loss', train_loss, epoch)

            # Validate the model
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            interpretable_weights= []
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output, weights = model(data)
                    interpretable_weights.append(weights)
                    loss = criterion(output, data.y)
                    val_loss += loss.item()
                    preds = output.argmax(dim=1)
                    val_correct += (preds == data.y).sum().item()
                    val_total += data.y.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(data.y.cpu().numpy())
            

            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            # Compute additional metrics using scikit-learn
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)  # weighted for class imbalance
            val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(val_labels, val_preds)


            writer.add_scalar(f'Fold_{fold+1}/Metrics/Val_Loss', val_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Metrics/Val_Accuracy', val_accuracy, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Metrics/Val_F1', val_f1, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Metrics/Val_Precision', val_precision, epoch)
            

            print_with_timestamp(f"Epoch {epoch + 1}/{epochs}\t||\tTrain Loss: {train_loss:.4f}\t|\tVal Loss: {val_loss:.4f}\t|\tAccuracy: {val_accuracy:.4f}\t|\tF1-Score: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print_with_timestamp(f"Early stopping at epoch {epoch + 1}")
                    break
            # Early stopping check
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_f1 = val_f1
                best_val_precision = val_precision
                best_confusion_matrix = conf_matrix
            

        
        print_with_timestamp(f"Fold {fold +1} Best Validation Loss: {best_val_loss:.4f}")
        print_with_timestamp(f"Fold {fold +1} Best Metrics: Accuracy: {best_val_acc:.4f}, Precision: {best_val_precision:.4f}, F1 Score: {best_val_f1:.4f}")
        best_conf_matrix_str = np.array2string(best_confusion_matrix, separator=' ', max_line_width=np.inf).replace('\n', ' ')
        print_with_timestamp(f"Fold {fold +1} Best Confusion Matrix: {best_conf_matrix_str}")

        # Store metrics for the current fold
        all_fold_metrics['accuracy'].append(best_val_acc)
        all_fold_metrics['precision'].append(best_val_precision)
        all_fold_metrics['f1'].append(best_val_f1)
        all_fold_metrics['conf_matrix'].append(conf_matrix)
        all_fold_metrics['val_loss'].append(best_val_loss)
        all_fold_metrics['weights'].append(torch.cat(interpretable_weights).mean(dim=0))

        # Print average metrics until the latest fold
        avg_accuracy = np.mean(all_fold_metrics['accuracy'])
        avg_precision = np.mean(all_fold_metrics['precision'])
        avg_f1 = np.mean(all_fold_metrics['f1'])
        avg_val_loss = np.mean(all_fold_metrics['val_loss'])
        
        print_with_timestamp(f"Average Metrics until Fold {fold + 1}: Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, F1 Score: {avg_f1:.4f}")
        print_with_timestamp(f"Average Lowest Validation Loss until Fold {fold + 1}: {avg_val_loss:.4f}")

    # Compute and print final metrics after cross-validation
    final_accuracy = np.mean(all_fold_metrics['accuracy'])
    final_precision = np.mean(all_fold_metrics['precision'])
    final_f1 = np.mean(all_fold_metrics['f1'])
    

    print_with_timestamp("Training completed.")
    print_with_timestamp(f"Final Metrics | Accuracy: {final_accuracy:.4f} | Precision: {final_precision:.4f} | F1 Score: {final_f1:.4f}")

    # Print the average confusion matrix
    avg_conf_matrix = np.mean(all_fold_metrics['conf_matrix'], axis=0)
    avg_conf_matrix_str = np.array2string(avg_conf_matrix, separator=' ', max_line_width=np.inf).replace('\n', ' ')
    print_with_timestamp(f"Final Confusion Matrix: {avg_conf_matrix_str}")


    final_weights = torch.stack(all_fold_metrics['weights']).mean(dim=0)
    print_with_timestamp(f"Final Weights: {final_weights}")
    # Define class names based on your specific classes
    if args.dataset == 'ppmi':
        class_names = ['Control', 'Prodromal', 'Patient', 'Swedd']
    else:
        class_names = ['Control', 'Patient']
    # Create a confusion matrix plot
    cm_figure = plot_confusion_matrix(avg_conf_matrix, class_names)
    # Convert the plot to a TensorBoard image
    cm_image = plot_to_image(cm_figure)
    # Ensure the image is in the correct format
    if cm_image.dtype != np.uint8:
        cm_image = (cm_image * 255).astype(np.uint8)
    writer.add_image("Confusion Matrix", cm_image, global_step=0, dataformats='HWC')
    # Close the TensorBoard writer
    writer.close()