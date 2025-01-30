import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# --------------------------
# 1. Configuration
# --------------------------
SEED = 42
NUM_TASKS = 5
CLASSES_PER_TASK = 20
BATCH_SIZE = 256
EPOCHS = 10
SPARSITY = 0.3  # Reduced from 0.5 to retain more plasticity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

# --------------------------
# 2. Model Architecture
# --------------------------
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*8*8, 100)
        
        # Initialize all masks to 1 (plastic)
        self.mask1 = nn.Parameter(torch.ones_like(self.conv1.weight), requires_grad=False)
        self.mask2 = nn.Parameter(torch.ones_like(self.conv2.weight), requires_grad=False)
        self.mask_fc = nn.Parameter(torch.ones_like(self.fc.weight), requires_grad=False)

    def apply_mask(self, task_masks):
        """Apply masks from a specific task"""
        self.mask1.copy_(task_masks['conv1.weight'])
        self.mask2.copy_(task_masks['conv2.weight'])
        self.mask_fc.copy_(task_masks['fc.weight'])

    def forward(self, x):
        conv1_weight = self.conv1.weight * self.mask1
        x = F.relu(self.bn1(F.conv2d(x, conv1_weight, self.conv1.bias, padding=1)))
        x = F.max_pool2d(x, 2)
        
        conv2_weight = self.conv2.weight * self.mask2
        x = F.relu(self.bn2(F.conv2d(x, conv2_weight, self.conv2.bias, padding=1)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        fc_weight = self.fc.weight * self.mask_fc
        x = F.linear(x, fc_weight, self.fc.bias)
        return x

# --------------------------
# 3. Dynamic Plasticity Controller
# --------------------------
class PlasticityController(nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_params, 512),
            nn.ReLU(),
            nn.Linear(512, num_params),
            nn.Sigmoid()
        )
        
    def forward(self, grad_norms):
        return self.net(grad_norms)

# --------------------------
# 4. Training Utilities
# --------------------------
def split_cifar100_by_tasks(dataset, task_id):
    classes = list(range(task_id*20, (task_id+1)*20))
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

def compute_fisher(model, dataloader):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2)
    for n in fisher:
        fisher[n] /= len(dataloader.dataset)
    return fisher

def evaluate(model, test_loaders, task_masks, current_task):
    model.eval()
    accuracies = []
    for task in range(current_task + 1):
        # Load masks from when this task was trained
        model.apply_mask(task_masks[task])
        
        correct = 0
        total = 0
        for x, y in test_loaders[task]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        accuracies.append(acc)
        print(f"Task {task+1} Accuracy: {acc:.2f}%")
    avg_acc = np.mean(accuracies)
    print(f"Average Accuracy after Task {current_task+1}: {avg_acc:.2f}%")
    return avg_acc

# --------------------------
# 5. Main Training Loop (Fixed)
# --------------------------
def train_nse():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    test_loaders = [
        DataLoader(split_cifar100_by_tasks(test_dataset, task), batch_size=BATCH_SIZE, shuffle=False)
        for task in range(NUM_TASKS)
    ]

    model = BaseCNN().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    plast_controller = PlasticityController(num_params).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(plast_controller.parameters()), lr=1e-3)
    
    # Store masks for each task
    task_masks = []

    for task in range(NUM_TASKS):
        print(f"\n=== Training Task {task+1}/{NUM_TASKS} ===")
        train_subset = split_cifar100_by_tasks(train_dataset, task)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        for epoch in range(EPOCHS):
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                loss = F.cross_entropy(output, y)
                optimizer.zero_grad()
                loss.backward()

                grad_norms = torch.cat([p.grad.view(-1).abs() for p in model.parameters() if p.grad is not None])
                if grad_norms.numel() > 0:
                    plasticity = plast_controller(grad_norms.unsqueeze(0)).squeeze(0)
                    ptr = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            p_size = p.numel()
                            p.grad *= plasticity[ptr:ptr+p_size].view_as(p.grad)
                            ptr += p_size
                optimizer.step()

        # Compute and store masks for this task
        fisher = compute_fisher(model, train_loader)
        current_masks = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher:
                    importance = fisher[name]
                    k = int(SPARSITY * param.numel())
                    _, idx = torch.topk(importance.flatten(), k, largest=False)
                    mask = torch.ones_like(param)
                    mask.view(-1)[idx] = 0
                    
                    if name == "conv1.weight":
                        current_masks[name] = mask.clone()
                        model.mask1.copy_(mask)
                    elif name == "conv2.weight":
                        current_masks[name] = mask.clone()
                        model.mask2.copy_(mask)
                    elif name == "fc.weight":
                        current_masks[name] = mask.clone()
                        model.mask_fc.copy_(mask)
        
        task_masks.append(current_masks)
        
        # Evaluation with task-specific masks
        print("\n=== Evaluation ===")
        avg_acc = evaluate(model, test_loaders, task_masks, task)
        
# --------------------------
# 6. Run the Experiment
# --------------------------
if __name__ == "__main__":
    train_nse()
