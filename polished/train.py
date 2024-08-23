
import torch
import numpy as np
import datetime

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from dataset import collate_function
from models import get_model

import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image

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

def train(model_name, dataset, device, args):
    # Hyperparameters
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
    test_size = args.test_size
    n_nodes = 116
    edge_dim = dataset.edge_features_count()
    out_dim = 4 if args.dataset == 'ppmi' else 2
    # Cross-validation setup
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # Initialize lists to store metrics for all folds
    all_fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'conf_matrix':[]}


    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'polished/runs/{model_name}_{args.dataset}_s{seed}_f{n_folds}_e{epochs}_bs{batch_size}_lr{learning_rate}_hd{hidden_dim}_d{dropout}_h{heads}_l{n_layers}_ts{test_size}_a{args.augmented}_{timestamp}')
    writer.add_text('Arguments', str(args))

    generator = torch.Generator(device=device).manual_seed(seed)
    # Training and evaluation loop
    for fold, (train_index, test_index) in enumerate(kf.split(dataset, dataset.labels.to('cpu'))):
        print_with_timestamp(f"Fold {fold + 1}/{n_folds}")
        train_data = torch.utils.data.Subset(dataset, train_index)
        test_data = torch.utils.data.Subset(dataset, test_index)
        train_indices, val_indices = train_test_split(range(len(train_data)), test_size=test_size, random_state=seed)
        train_subset = torch.utils.data.Subset(train_data, train_indices)
        val_subset = torch.utils.data.Subset(train_data, val_indices)

        # Print label distributions
        train_labels = [dataset.labels[idx].item() for idx in train_index]
        val_labels = [dataset.labels[train_index[idx]].item() for idx in val_indices]
        test_labels = [dataset.labels[idx].item() for idx in test_index]
        
        print_with_timestamp(f"Training labels distribution: {np.bincount(train_labels)}")
        print_with_timestamp(f"Validation labels distribution: {np.bincount(val_labels)}")
        print_with_timestamp(f"Test labels distribution: {np.bincount(test_labels)}")

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
        train_loader.collate_fn = collate_function
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, generator=generator)
        val_loader.collate_fn = collate_function
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, generator=generator)
        test_loader.collate_fn = collate_function

        model = get_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience//3)
        fold_gradients = {}

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(dataset.labels.to('cpu').numpy()), y=dataset.labels.to('cpu').numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print_with_timestamp(f'Class weights: {class_weights}')
        print_with_timestamp(f"Epoch/Loss\t||\tTraining\t|\tValidation\t")
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

        best_val_loss = float('inf')
        patience_counter = 0

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
                output = model(data)
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
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output = model(data)
                    loss = criterion(output, data.y)
                    val_loss += loss.item()
                    preds = output.argmax(dim=1)
                    val_correct += (preds == data.y).sum().item()
                    val_total += data.y.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(data.y.cpu().numpy())

            val_loss /= len(val_loader)
            writer.add_scalar(f'Fold_{fold+1}/Metrics/Val_Loss', val_loss, epoch)

            print_with_timestamp(f"{epoch + 1}/{epochs}\t\t||\t{train_loss:.4f}\t\t|\t{val_loss:.4f}\t")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print_with_timestamp(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load the best model state
        model.load_state_dict(best_model_state)
        
        # Evaluate the model on the test set
        model.eval()
        test_preds = []
        test_labels = []

        for data in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
                preds = output.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(data.y.cpu().numpy())


        # Compute metrics
        acc = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, test_preds, average='weighted')

        # Confusion matrix for the test set
        conf_matrix = confusion_matrix(test_labels, test_preds)

        writer.add_scalar(f'Fold_{fold+1}/Test_Accuracy', acc, epoch)
        writer.add_scalar(f'Fold_{fold+1}/Test_Precision', precision, epoch)
        writer.add_scalar(f'Fold_{fold+1}/Test_Recall', recall, epoch)
        writer.add_scalar(f'Fold_{fold+1}/Test_F1', f1, epoch)
        print_with_timestamp(f"Fold {fold + 1} Metrics: Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Store metrics for the current fold
        all_fold_metrics['accuracy'].append(acc)
        all_fold_metrics['precision'].append(precision)
        all_fold_metrics['recall'].append(recall)
        all_fold_metrics['f1'].append(f1)
        all_fold_metrics['conf_matrix'].append(conf_matrix)

        # Print average metrics until the latest fold
        avg_accuracy = np.mean(all_fold_metrics['accuracy'])
        avg_precision = np.mean(all_fold_metrics['precision'])
        avg_recall = np.mean(all_fold_metrics['recall'])
        avg_f1 = np.mean(all_fold_metrics['f1'])
        print_with_timestamp(f"Average Metrics until Fold {fold + 1}: Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")

    # Compute and print final metrics after cross-validation
    final_accuracy = np.mean(all_fold_metrics['accuracy'])
    final_precision = np.mean(all_fold_metrics['precision'])
    final_recall = np.mean(all_fold_metrics['recall'])
    final_f1 = np.mean(all_fold_metrics['f1'])

    print_with_timestamp("Training completed.")
    print_with_timestamp(f"Final Metrics | Accuracy: {final_accuracy:.4f} | Precision: {final_precision:.4f} | Recall: {final_recall:.4f} | F1 Score: {final_f1:.4f}")

    # Print the average confusion matrix
    avg_conf_matrix = np.mean(all_fold_metrics['conf_matrix'], axis=0)
    print_with_timestamp(f"Final Confusion Matrix:\n{avg_conf_matrix}")
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