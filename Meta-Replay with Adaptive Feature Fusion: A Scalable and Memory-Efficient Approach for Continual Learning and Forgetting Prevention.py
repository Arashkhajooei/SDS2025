import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-3
EPOCHS_PER_TASK = 5
NUM_TASKS = 5
CLASSES_PER_TASK = 2
IMG_SIZE = 28
LATENT_DIM = 128  # Feature Fusion Replay latent space
EMBEDDING_DIM = 64  # Task Embedding Memory Module

# MNIST Task Division (5 tasks, 2 classes each)
TASKS = [[0,1], [2,3], [4,5], [6,7], [8,9]]

# 1. Dataset Wrapper
class TaskDataset(Dataset):
    def __init__(self, base_dataset, task_classes):
        self.data = []
        self.labels = []
        class_map = {cls: idx for idx, cls in enumerate(task_classes)}
        
        for img, lbl in base_dataset:
            if lbl in task_classes:
                self.data.append(img)
                self.labels.append(class_map[lbl])
        
        self.data = torch.stack(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

# 2. Task Embedding Memory Module (TEMM)
class TaskEmbeddingMemory(nn.Module):
    def __init__(self, num_tasks, embedding_dim):
        super(TaskEmbeddingMemory, self).__init__()
        self.embeddings = nn.Embedding(num_tasks, embedding_dim)

    def forward(self, task_id):
        return self.embeddings(task_id)  # Shape: [batch_size, embedding_dim]

# 3. Feature Fusion Replay (FFR) - Autoencoder for compressed task replay
class FeatureFusionReplay(nn.Module):
    def __init__(self, latent_dim):
        super(FeatureFusionReplay, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * IMG_SIZE * IMG_SIZE, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * IMG_SIZE * IMG_SIZE),
            nn.ReLU(),
            nn.Unflatten(1, (64, IMG_SIZE, IMG_SIZE)),
            nn.ConvTranspose2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z)).view(-1, 1, IMG_SIZE, IMG_SIZE)

# 4. Meta-Replay Selector
class MetaReplaySelector(nn.Module):
    def __init__(self, embedding_dim):
        super(MetaReplaySelector, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)  # Predicts replay importance

    def forward(self, task_embedding):
        return torch.sigmoid(self.fc(task_embedding))  # Output: Probability of selecting replay samples

# 5. Continual Learning Model
class ContinualLearningNet(nn.Module):
    def __init__(self, num_tasks):
        super(ContinualLearningNet, self).__init__()
        self.features = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.current_channels = 1
        self.spatial_size = IMG_SIZE
        self.task_embedding_module = TaskEmbeddingMemory(num_tasks, EMBEDDING_DIM).to(DEVICE)

    def add_task(self, task_id):
        out_channels = min(32 * (2 ** task_id), 128)
        apply_pool = task_id < 3
        
        block = nn.Sequential(
            nn.Conv2d(self.current_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if apply_pool else nn.Identity()
        ).to(DEVICE)
        self.features.append(block)

        if apply_pool:
            self.spatial_size = self.spatial_size // 2
        self.current_channels = out_channels

        self.heads.append(nn.Linear(out_channels * (self.spatial_size ** 2), CLASSES_PER_TASK).to(DEVICE))

    def forward_features(self, x, task_id=None):
        if task_id is None:
            task_id = len(self.features) - 1
        for i in range(task_id + 1):
            x = self.features[i](x)
        return x.view(x.size(0), -1)

    def forward(self, x, task_id):
        x = self.forward_features(x, task_id)
        return self.heads[task_id](x)

# 6. Evaluation Function
def evaluate_model(model, task_id, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x, task_id)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# 7. Training Framework
class CLTrainerMRDTE:
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.full_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.model = ContinualLearningNet(NUM_TASKS).to(DEVICE)
        self.feature_replay = FeatureFusionReplay(LATENT_DIM).to(DEVICE)
        self.meta_replay = MetaReplaySelector(EMBEDDING_DIM).to(DEVICE)

    def train_task(self, task_id):
        print(f"\n=== Training Task {task_id+1} ===")
        task_data = TaskDataset(self.full_data, TASKS[task_id])
        task_loader = DataLoader(task_data, batch_size=BATCH_SIZE, shuffle=True)
        self.model.add_task(task_id)
        optimizer = optim.Adam(self.model.parameters(), lr=LR)

        for epoch in range(EPOCHS_PER_TASK):
            total_loss = 0.0
            for x, y in tqdm(task_loader, desc=f"Epoch {epoch+1}/{EPOCHS_PER_TASK}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(x, task_id)
                loss = F.cross_entropy(outputs, y)

                if task_id > 0:
                    task_embedding = self.model.task_embedding_module(torch.tensor([task_id], device=DEVICE))
                    replay_prob = self.meta_replay(task_embedding).item()
                    if random.random() < replay_prob:
                        replay_z = torch.randn((BATCH_SIZE, LATENT_DIM), device=DEVICE)
                        replay_x = self.feature_replay.decode(replay_z)
                        replay_x = self.model.forward_features(replay_x, task_id - 1)
                        replay_y = torch.randint(0, CLASSES_PER_TASK, (BATCH_SIZE,), device=DEVICE)
                        loss += 0.5 * F.cross_entropy(self.model.heads[task_id - 1](replay_x), replay_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Task {task_id+1} Epoch {epoch+1} Loss: {total_loss / len(task_loader):.4f}")
        self.evaluate_all_tasks()

    def evaluate_all_tasks(self):
        for t in range(len(self.model.heads)):
            test_loader = DataLoader(TaskDataset(self.test_data, TASKS[t]), batch_size=BATCH_SIZE, shuffle=False)
            acc = evaluate_model(self.model, t, test_loader)
            print(f"Task {t+1} Accuracy: {acc:.2f}%")

# Run Training & Evaluation
if __name__ == "__main__":
    trainer = CLTrainerMRDTE()
    for task in range(NUM_TASKS):
        trainer.train_task(task)
