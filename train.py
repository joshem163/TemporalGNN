import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import statistics
import argparse
import sys
from model import CNNTransformer
from modules import *
from data_loader import load_data_feature

# Argument parsing
#sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


def main():
    data_names = ["infectious_ct1", "dblp_ct1", "facebook_ct1", "tumblr_ct1"]
    for dataset_name in data_names:
        print(f'\n=== Running on dataset: {dataset_name} ===')
        X0,y0= load_data_feature(dataset_name)
        X = torch.tensor(X0, dtype=torch.float32)
        y = torch.tensor(y0, dtype=torch.long)

        num_classes = len(torch.unique(y))
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

        acc_per_fold = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n🌀 Fold {fold}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32)

            # model = SimpleViT(patch_dim=4, num_patches=200, num_classes=num_classes).to(device)
            model = CNNTransformer(num_classes=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            # Lists to store metrics
            train_losses = []
            train_accuracies = []
            test_accuracies = []

            for epoch in tqdm(range(1, args.epochs + 1), desc=f"Epochs (Fold {fold})"):
                # Train
                model.train()
                correct_train = 0
                total_train = 0
                epoch_train_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    output = model(xb)
                    loss = criterion(output, yb)
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()

                    pred = output.argmax(dim=1)
                    correct_train += (pred == yb).sum().item()
                    total_train += yb.size(0)
                avg_train_loss = epoch_train_loss / len(train_loader)
                train_acc = correct_train / total_train
                train_losses.append(avg_train_loss)
                train_accuracies.append(train_acc)

                # Validation
                model.eval()
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        output = model(xb)
                        pred = output.argmax(dim=1)
                        correct_val += (pred == yb).sum().item()
                        total_val += yb.size(0)

                val_acc = correct_val / total_val
                test_accuracies.append(val_acc)
                #tqdm.write(
                 #   f"Epoch {epoch}:,Train loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

            # acc_per_fold.append(val_acc)
            #         print(f'Score for fold {fold_no}: ')
            accuracy = print_stat(train_accuracies, test_accuracies)
            acc_per_fold.append(accuracy[0])
        print(f'=== Final Results for {dataset_name} ===')
        print("\n📊 Final Cross-Validation Results:")
        print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in acc_per_fold]}")
        print(f"Average Accuracy: {np.mean(acc_per_fold):.4f} ± {np.std(acc_per_fold):.4f}")


if __name__ == "__main__":
    main()