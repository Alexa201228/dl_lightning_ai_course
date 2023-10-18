import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools import compute_accuracy, get_dataset_loaders, PyTorchMLP


def compute_total_accuracy(model, dataloader, device=None):
    """
    Evaluation of trained model
    :param model: Trained model
    :param dataloader: PyTorch dataloader
    :param device: name of device to evaluate on
    :return: loss / number of examples
    """

    if device is not None:
        device = torch.device(device)

    model = model.eval()
    loss = 0.0
    examples = 0.0

    for idx, (features, labels) in enumerate(dataloader):

        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
            batch_loss = F.cross_entropy(logits, labels, reduction="sum")

        loss += batch_loss.item()
        examples += logits.shape[0]


    return loss / examples


def train_model(model: torch.nn.Module, optimizer: torch.optim, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 10, seed: int = 1, device = None):
    """

    :param model:
    :param optimizer:
    :param train_loader:
    :param val_loader:
    :param num_epochs:
    :param seed:
    :param device:
    :return:
    """

    if device is not None:
        device = torch.device(device)

    torch.manual_seed(seed)

    for epoch in range(num_epochs):
        model = model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):

            features, labels = features.to(device), labels.to(device)

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 250:

                val_loss = compute_total_accuracy(model, val_loader, device=device)
                print(f"Epoch: {epoch + 1:03d} / {num_epochs:03d}"
                      f" | Batch {batch_idx:03d} / {len(train_loader):03d}"
                      f" | Train loss: {loss:2f}"
                      f" | Val total loss: {val_loss:.4f}")


if __name__ == "__main__":

    print(f"Torch CUDA is available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataset_loaders()

    model = PyTorchMLP(num_features=784, num_classes=10)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train_model(model, optimizer, train_loader, val_loader, num_epochs=10, seed=1, device=device)

    train_acc = compute_accuracy(model, train_loader, device)
    val_acc = compute_accuracy(model, val_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)

    print(f"Train acc: {train_acc * 100:.2f}%")
    print(f"Val acc: {val_acc * 100:.2f}%")
    print(f"Test acc: {test_acc * 100:.2f}%")

