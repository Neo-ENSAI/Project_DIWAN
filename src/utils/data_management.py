from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from tqdm.auto import tqdm
from IPython.display import display, clear_output
from matplotlib.ticker import MaxNLocator
import numpy as np


class LiveLossPlot():
    """Refreshes a matplotlib plot of the training loss, validation loss,
    and validation accuracy in place, once per epoch."""

    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    def update(self, epoch, train_loss, val_loss, val_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        self.ax_loss.clear()
        self.ax_loss.plot(self.epochs, self.train_losses, label="Train")
        self.ax_loss.plot(self.epochs, self.val_losses, label="Val")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_loss.legend()

        self.ax_acc.clear()
        self.ax_acc.plot(self.epochs, self.val_accuracies, color="green")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Val accuracy")
        self.ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))

        clear_output(wait=True)
        display(self.fig)



class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ls_img = np.array(self.X[idx, 0], dtype=np.float32)
        ct_img = np.array(self.X[idx, 1], dtype=np.float32)
        raw_y = self.y[idx]
        target_img = np.array(raw_y, dtype=np.float32)

        # mise en forme demandé par Unet (2, H, W)
        x_sample = torch.tensor(np.stack([ct_img, ls_img], axis=0), dtype=torch.float32)

        # forme (1, H, W)
        y_sample = torch.tensor(target_img, dtype=torch.float32)
        if len(y_sample.shape) == 2:
            y_sample = y_sample.unsqueeze(0)

        return x_sample, y_sample



class Trainer:
    def __init__(self, model, optimizer, criterion, max_epochs, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.device = device

    def fit(self, train_dataloader, val_dataloader):
        for epoch in range(self.max_epochs):
            # train
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.max_epochs} [Train]')
            for X, y in pbar:
                X = X.to(self.device)
                y = y.to(self.device)

                # Passe avant (Forward)
                preds = self.model(X)
                loss = self.criterion(preds, y)

                # Rétropropagation (Backward) & Optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            avg_train_loss = train_loss / len(train_dataloader)

            # val
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X, y in val_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    preds = self.model(X)
                    loss = self.criterion(preds, y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)

            print(f"--> Epoch {epoch+1} terminée | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        print("Entraînement terminé avec succès !")


def evaluate(dataloader, model, device, set_='Test'):
    model.eval()
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        batch_loss, pred, y_true = model.validation_step((X, y))
        loss += batch_loss
        correct += (pred == y_true).type(torch.float).sum().item()

    loss /= size
    accuracy = correct / size
    print(f"\n{set_} set: \n  Accuracy: {(100 * accuracy): >0.1f}%, Avg loss: {loss: >8f} \n")
    return loss, accuracy
