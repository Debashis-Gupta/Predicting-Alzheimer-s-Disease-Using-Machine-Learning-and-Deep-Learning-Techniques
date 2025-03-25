import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.models as models

# Dataset class for tabular data
class AlzheimerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ResNet-50 adapted for tabular data
class ResNet50ForTabular(nn.Module):
    def __init__(self, input_size):
        super(ResNet50ForTabular, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.input_projection = nn.Linear(input_size, 3 * 224 * 224)  # Project tabular data to 3x224x224
        self.reshape = lambda x: x.view(-1, 3, 224, 224)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 2)  # Binary classification
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.reshape(x)
        x = self.resnet50(x)
        return x

# Training function for a single fold
def train_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

# Evaluation function for a single fold
def evaluate_fold(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_labels, all_probs

# Main training function with k-fold cross-validation
def train_resnet_network(X, y, outdir_path, batch_size=32, epochs=50, learning_rate=0.01, k_folds=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_histories = []
    fold_metrics = {'accuracy': [], 'roc_auc': []}
    fold_conf_matrices = []
    all_fpr = []
    all_tpr = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\nFold {fold+1}/{k_folds}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = AlzheimerDataset(X_train, y_train)
        val_dataset = AlzheimerDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        input_size = X.shape[1]
        model = ResNet50ForTabular(input_size).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        # Train the model for this fold
        history = train_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs)
        fold_histories.append(history)
        
        # Evaluate the fold
        preds, labels, probs = evaluate_fold(model, val_loader, device)
        
        accuracy = accuracy_score(labels, preds)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(labels, preds)
        
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['roc_auc'].append(roc_auc)
        fold_conf_matrices.append(cm)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        # Plot ROC curve for this fold
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold+1}')
        plt.legend(loc="lower right")
        plt.savefig(f"{outdir_path}/roc_curve_fold_{fold+1}.jpeg", dpi=300)
        plt.close()
    
    # Compute mean and std across folds
    mean_metrics = {key: np.mean(val) for key, val in fold_metrics.items()}
    std_metrics = {key: np.std(val) for key, val in fold_metrics.items()}
    
    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Mean ROC-AUC: {mean_metrics['roc_auc']:.4f} ± {std_metrics['roc_auc']:.4f}")
    
    # Aggregate confusion matrices
    mean_cm = np.mean(fold_conf_matrices, axis=0)
    print("\nMean Confusion Matrix:")
    print(mean_cm)
    
    # Plot mean training curves with std deviation
    epochs_range = range(epochs)
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    
    for metric in metrics:
        all_curves = np.array([h[metric] for h in fold_histories])
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        plt.figure()
        plt.plot(epochs_range, mean_curve, label=f'Mean {metric.replace("_", " ").title()}')
        plt.fill_between(epochs_range, mean_curve - std_curve, mean_curve + std_curve, 
                        alpha=0.3, label='±1 Std Dev')
        plt.title(f'{metric.replace("_", " ").title()} Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.savefig(f"{outdir_path}/{metric}_curve.jpeg", dpi=300)
        plt.close()
    
    # Plot mean ROC curve with std (interpolated to common FPR points)
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)
    
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_metrics["roc_auc"]:.4f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.3, label='±1 Std Dev')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve Across Folds')
    plt.legend(loc="lower right")
    plt.savefig(f"{outdir_path}/mean_roc_curve.jpeg", dpi=300)
    plt.close()
    
    # Save the final model (from the last fold)
    torch.save(model.state_dict(), f"{outdir_path}/alzheimer_resnet50_model.pth")
    print(f"Final model saved at: {outdir_path}/alzheimer_resnet50_model.pth")
    
    return model, fold_histories, fold_metrics

# Prediction function
def predict_with_resnet_network(model, X, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()

# # Example usage
# if __name__ == "__main__":
    
#     np.random.seed(42)
   
#     outdir_path = "resnet_result"
#     import os
#     os.makedirs(outdir_path, exist_ok=True)
    
#     model, histories, metrics = train_neural_network_resnet(
#         X, y, outdir_path, batch_size=32, epochs=5, learning_rate=0.01, k_folds=5
#     )
    
#     # Example prediction
#     X_test = np.random.rand(10, 10)  # 10 test samples
#     preds, probs = predict_with_neural_network_resnet(model, X_test)
#     print("\nPredictions:", preds)
#     print("Probabilities:", probs)