import torch
from tqdm.auto import tqdm


def count_correct_predictions(
    predictions:torch.Tensor,
    labels:torch.Tensor,
) -> int:
    "Count how many predictions match corresponding labels."
    return predictions.argmax(dim=1).eq(labels).sum().item()


def train_step(
    device:torch.device,
    loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LRScheduler,
    epoch:int,
    onecyclelr:bool,
) -> tuple[float, float]:
    "Update weights and compute average training loss and accuracy for one epoch."
    model.train()
    train_loss = 0
    correct = 0
    processed = 0
    epoch += 1
    pbar = tqdm(loader)
    for batch, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if onecyclelr: scheduler.step()
        train_loss += loss.item()
        correct += count_correct_predictions(pred, y)
        processed += len(X)
        pbar.set_description(desc=(
            f"Epoch = {epoch}, "
            f"Batch = {batch}, "
            f"Loss = {loss.item():0.4f}, "
            f"Accuracy = {correct/processed:0.2%}"
        ))
    if not onecyclelr: scheduler.step()
    n = len(loader.dataset)
    train_loss /= n
    train_acc = correct / n
    print(
        f"Train: "
        f"Loss = {train_loss:0.5f}, "
        f"Accuracy = {train_acc:0.2%}, "
        f"Epoch = {epoch}"
    )
    return train_loss, train_acc


def test_step(
    device:torch.device,
    loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.Module,
) -> tuple[float, float]:
    "Compute average test loss and accuracy for one epoch."
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            test_loss += loss.item()
            correct += count_correct_predictions(pred, y)
    n = len(loader.dataset)
    test_loss /= n
    test_acc = correct / n
    print(
        f"Test : "
        f"Loss = {test_loss:.5f}, "
        f"Accuracy = {test_acc:.2%}\n"
    )
    return test_loss, test_acc


def train(
    device:torch.device,
    train_loader:torch.utils.data.DataLoader,
    test_loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LRScheduler,
    epochs:int,
    onecyclelr:bool=True,
) -> dict[str, list[float]]:
    "Train and test for given epochs and return the results."
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            device,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            onecyclelr=onecyclelr,
        )
        test_loss, test_acc = test_step(
            device,
            test_loader,
            model,
            criterion,
        )
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc*100)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc*100)
    return results