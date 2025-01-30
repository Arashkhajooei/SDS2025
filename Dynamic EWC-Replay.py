import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

################################################################################
# 1. Configuration and Hyperparameters
################################################################################

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
LR = 0.001
EPOCHS_PER_TASK = 100
LAMBDA_EWC = 2000.0      # Base EWC weight
REPLAY_BUFFER_SIZE = 5000
REPLAY_BATCH_RATIO = 1.0
SUBSET_COUNT_PER_TASK = 400

NUM_TASKS = 5
CLASSES_PER_TASK = 20

# CIFAR-100 is split into 5 tasks, each with 20 classes
TASKS = [
    list(range(0, 20)),
    list(range(20, 40)),
    list(range(40, 60)),
    list(range(60, 80)),
    list(range(80, 100))
]

################################################################################
# 2. Dataset and DataLoader for Incremental Tasks
################################################################################

class LocalLabelDataset(Dataset):
    """
    Wraps a subset of CIFAR-100 data and remaps global labels [0..99]
    to local labels [0..(len(class_list)-1)].
    """
    def __init__(self, base_dataset, indices, class_list):
        super().__init__()
        self.subset = Subset(base_dataset, indices)
        # Mapping from global label to local label
        self.class_to_local = {cls: i for i, cls in enumerate(class_list)}

    def __getitem__(self, idx):
        img, global_label = self.subset[idx]
        local_label = self.class_to_local[global_label]
        return img, local_label

    def __len__(self):
        return len(self.subset)

def get_task_dataloader(dataset, class_list, batch_size=BATCH_SIZE, train=True):
    """
    Creates a DataLoader for a subset of classes in CIFAR-100,
    remapping global labels to local labels for each task.
    """
    indices = [i for i, (_, lbl) in enumerate(dataset) if lbl in class_list]
    local_dataset = LocalLabelDataset(dataset, indices, class_list)
    loader = DataLoader(
        local_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,      # Avoid multiprocessing issues
        pin_memory=False
    )
    return loader

################################################################################
# 3. Multi-Head CNN Model
################################################################################

class MultiHeadCNN(nn.Module):
    """
    Shared feature extractor + multiple task-specific heads.
    Each head has CLASSES_PER_TASK outputs.
    """
    def __init__(self, in_channels=3, num_tasks=NUM_TASKS, classes_per_task=CLASSES_PER_TASK):
        super(MultiHeadCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # One linear "head" per task
        self.heads = nn.ModuleList([
            nn.Linear(512, classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x, task_idx):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.heads[task_idx](feats)

################################################################################
# 4. EWC Implementation
################################################################################

class EWC:
    """
    Stores the parameter values and Fisher information for a task,
    then penalizes changes to critical parameters for that task.
    """
    def __init__(self, model, dataloader, task_idx, device=DEVICE):
        self.model = model
        self.device = device
        self.task_idx = task_idx

        # Store model parameters after finishing the task
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        # Compute Fisher
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        self.model.eval()
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        total_samples = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()

            output = self.model(data, self.task_idx)
            loss = F.cross_entropy(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad.data.pow(2)

            total_samples += len(target)

        # Average over dataset
        for n in fisher_dict:
            fisher_dict[n] /= total_samples

        return fisher_dict

    def penalty(self, model):
        """
        EWC penalty for deviating from stored params, weighted by Fisher info.
        """
        loss = 0.0
        for n, p in model.named_parameters():
            # (p - old_param)^2 * fisher
            _loss = self.fisher[n] * (p - self.params[n]).pow(2)
            loss += _loss.sum()
        return loss

################################################################################
# 5. Replay Buffer
################################################################################

class ReplayBuffer:
    """
    Stores examples from previous tasks for replay.
    (x, y, task_id) with local labels (y).
    """
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = []
        self.max_size = max_size

    def add_samples(self, x, y, t):
        """
        x, y: CPU or GPU Tensors
        t: single integer or list of task IDs
        """
        if isinstance(t, int):
            t = [t]*len(x)
        for i in range(len(x)):
            self.buffer.append((x[i].cpu(), y[i].cpu(), t[i]))
        # Keep within max size
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def get_samples(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        x_list, y_list, t_list = [], [], []
        for idx in indices:
            x_list.append(self.buffer[idx][0])
            y_list.append(self.buffer[idx][1])
            t_list.append(self.buffer[idx][2])
        x = torch.stack(x_list, dim=0)
        y = torch.tensor(y_list, dtype=torch.long)
        t = torch.tensor(t_list, dtype=torch.long)
        return x, y, t

################################################################################
# 6. Dynamic Weighting
################################################################################

def compute_dynamic_weights(task_loss, replay_loss, ewc_loss):
    """
    Balances the three loss terms: new-task, replay, and EWC penalty.
    Each is scaled by (loss_component / total_loss).
    """
    # Avoid division by zero
    total = task_loss + replay_loss + ewc_loss
    if total.item() < 1e-7:
        return 1.0, 0.0, 0.0

    alpha = task_loss / total
    beta  = replay_loss / total
    gamma = ewc_loss / total
    return alpha, beta, gamma

################################################################################
# 7. Training Function
################################################################################

def train_one_task(
    model,
    optimizer,
    task_loader,
    ewc_list,
    replay_buffer,
    current_task_idx,
    device=DEVICE,
    lambda_ewc=LAMBDA_EWC,
    epochs=EPOCHS_PER_TASK
):
    """
    Trains the model on the new task with:
    - Task loss (on new data)
    - Replay loss (on old samples)
    - EWC penalty
    Weighted dynamically each iteration.
    """
    model.train()
    for epoch in range(1, epochs+1):
        epoch_task_loss = 0.0
        epoch_replay_loss = 0.0
        epoch_ewc_loss = 0.0

        for data, target in task_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # 1) Task Loss (tensor)
            output = model(data, current_task_idx)
            task_loss = F.cross_entropy(output, target)

            # 2) Replay Loss (tensor accumulation -> final float)
            replay_loss_tensor = torch.zeros([], device=device)
            replay_data, replay_labels, replay_tasks = replay_buffer.get_samples(batch_size=len(data))
            if replay_data is not None:
                replay_data = replay_data.to(device)
                replay_labels = replay_labels.to(device)
                replay_tasks = replay_tasks.to(device)
                unique_tasks = torch.unique(replay_tasks)
                for t_idx in unique_tasks:
                    mask = (replay_tasks == t_idx)
                    sub_data = replay_data[mask]
                    sub_labels = replay_labels[mask]
                    out_replay = model(sub_data, t_idx.item())
                    replay_loss_tensor += F.cross_entropy(out_replay, sub_labels)

            # 3) EWC Loss
            ewc_tensor = torch.zeros([], device=device)
            if len(ewc_list) > 0:
                for ewc_obj in ewc_list:
                    ewc_tensor += ewc_obj.penalty(model)

            # 4) Dynamic Weighting
            alpha, beta, gamma = compute_dynamic_weights(task_loss, replay_loss_tensor, ewc_tensor)
            total_loss = alpha*task_loss + beta*replay_loss_tensor + gamma*(lambda_ewc*ewc_tensor)

            total_loss.backward()
            optimizer.step()

            # Convert each to float for logging
            epoch_task_loss    += task_loss.item()
            epoch_replay_loss += replay_loss_tensor.item()
            epoch_ewc_loss    += ewc_tensor.item()

        print(f"[Epoch {epoch}/{epochs}] "
              f"Task: {epoch_task_loss:.2f}, "
              f"Replay: {epoch_replay_loss:.2f}, "
              f"EWC: {epoch_ewc_loss:.2f}")

################################################################################
# 8. Evaluation Function
################################################################################

def evaluate(model, test_loaders, current_task_idx, device=DEVICE):
    """
    Evaluates the model on all tasks up to 'current_task_idx'
    using the corresponding heads for each task.
    """
    model.eval()
    accuracies = []
    for t_idx, loader in enumerate(test_loaders[:current_task_idx+1]):
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                out = model(data, t_idx)  # Use head t_idx
                pred = out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total   += len(target)
        acc = 100.0 * correct / total
        accuracies.append(acc)
        print(f"Task {t_idx+1} Test Accuracy: {acc:.2f}%")
    avg_acc = np.mean(accuracies)
    print(f"Average accuracy after Task {current_task_idx+1}: {avg_acc:.2f}%")
    return avg_acc

################################################################################
# 9. Main Script
################################################################################

def main():
    # Environment vars to reduce threading issues
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    print(f"Using device: {DEVICE}")

    # Prepare datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762))
    ])

    train_data = datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform_train)
    test_data  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loaders = [get_task_dataloader(train_data, class_list) for class_list in TASKS]
    test_loaders  = [get_task_dataloader(test_data,  class_list, train=False) for class_list in TASKS]

    # Initialize model, optimizer
    model = MultiHeadCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Replay buffer + EWC objects
    replay_buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)
    ewc_list = []

    # Train tasks sequentially
    for task_idx, loader in enumerate(train_loaders):
        print(f"\n=== Training on Task {task_idx+1}/{NUM_TASKS} ===")
        # Train
        train_one_task(
            model=model,
            optimizer=optimizer,
            task_loader=loader,
            ewc_list=ewc_list,
            replay_buffer=replay_buffer,
            current_task_idx=task_idx,
            device=DEVICE,
            lambda_ewc=LAMBDA_EWC,
            epochs=EPOCHS_PER_TASK
        )

        # Store exemplars for replay
        # (Simple approach: add a single pass or up to SUBSET_COUNT_PER_TASK)
        samples_added = 0
        for data, target in loader:
            # Limit how many from the new task to store
            if samples_added >= SUBSET_COUNT_PER_TASK:
                break
            replay_buffer.add_samples(data, target, task_idx)
            samples_added += len(data)

        # Build EWC object for this finished task
        # Use the same training data or a smaller subset to compute Fisher
        ewc_loader = get_task_dataloader(train_data, TASKS[task_idx], batch_size=64, train=True)
        ewc_obj = EWC(model, ewc_loader, task_idx, device=DEVICE)
        ewc_list.append(ewc_obj)

        # Evaluate up to current task
        print(f"\n=== Evaluation after Task {task_idx+1} ===")
        evaluate(model, test_loaders, task_idx, device=DEVICE)

if __name__ == '__main__':
    main()
