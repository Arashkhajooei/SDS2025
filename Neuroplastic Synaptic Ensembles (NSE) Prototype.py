import torch
import torch.nn as nn
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

# --------------------------
# 2. Model Architecture
# --------------------------
class BaseCNN(nn.Module):
    """Shared feature extractor with synaptic masks"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*8*8, 100)  # Output for all CIFAR-100 classes
        
        # Mask parameters (1 = plastic, 0 = frozen)
        self.mask1 = nn.Parameter(torch.ones_like(self.conv1.weight), requires_grad=False)
        self.mask2 = nn.Parameter(torch.ones_like(self.conv2.weight), requires_grad=False)
        self.mask_fc = nn.Parameter(torch.ones_like(self.fc.weight), requires_grad=False)

    def apply_mask(self, task_id):
        """Apply task-specific mask during forward pass (simplified)"""
        self.conv1.weight.data *= self.mask1
        self.conv2.weight.data *= self.mask2
        self.fc.weight.data *= self.mask_fc

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------------------
# 3. Dynamic Plasticity Controller
# --------------------------
class PlasticityController(nn.Module):
    """Predicts plasticity coefficients for parameters"""
    def __init__(self, num_params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_params, 128),
            nn.ReLU(),
            nn.Linear(128, num_params),
            nn.Sigmoid()  # Output [0,1] plasticity coefficients
        )
        
    def forward(self, grad_norms):
        return self.net(grad_norms)

# --------------------------
# 4. Training Utilities
# --------------------------
def split_cifar100_by_tasks(dataset, task_id):
    """Split CIFAR-100 into sequential tasks (20 classes per task)"""
    classes = list(range(task_id*20, (task_id+1)*20))
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

def compute_fisher(model, dataloader):
    """Approximate Fisher Information for synaptic importance"""
    fisher = {}
    model.eval()
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
        
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)
                
    # Normalize
    for name in fisher:
        fisher[name] /= len(dataloader.dataset)
    return fisher

# --------------------------
# 5. Main Training Loop
# --------------------------
def train_nse():
    # Load CIFAR-100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Initialize model and plasticity controller
    model = BaseCNN().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    plast_controller = PlasticityController(num_params).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(plast_controller.parameters()), lr=1e-3)
    
    # Store Fisher information and masks for each task
    task_masks = {}
    
    for task in range(NUM_TASKS):
        print(f"\n=== Training Task {task+1}/{NUM_TASKS} ===")
        
        # Prepare task data
        train_subset = split_cifar100_by_tasks(train_dataset, task)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train on current task
        model.train()
        for epoch in range(EPOCHS):
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # Forward with current masks
                model.apply_mask(task)
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                
                # Backward with plasticity control
                optimizer.zero_grad()
                loss.backward()
                
                # Get plasticity coefficients
                grad_norms = torch.cat([p.grad.view(-1).abs() for p in model.parameters()])
                plasticity = plast_controller(grad_norms.detach())
                
                # Apply plasticity to gradients
                ptr = 0
                for p in model.parameters():
                    p_size = p.numel()
                    p.grad *= plasticity[ptr:ptr+p_size].view_as(p.grad)
                    ptr += p_size
                
                optimizer.step()
        
        # Compute Fisher and update masks
        fisher = compute_fisher(model, train_loader)
        with torch.no_grad():
            # Freeze top 50% important parameters
            for name, param in model.named_parameters():
                importance = fisher[name]
                k = int(0.5 * param.numel())
                _, idx = torch.topk(importance.flatten(), k, largest=False)
                mask = torch.ones_like(param)
                mask.view(-1)[idx] = 0  # Frozen parameters
                
                # Store mask for this task
                if name == "conv1.weight": model.mask1.copy_(mask)
                elif name == "conv2.weight": model.mask2.copy_(mask)
                elif name == "fc.weight": model.mask_fc.copy_(mask)
                
        task_masks[task] = {name: mask.clone() for name, mask in model.named_buffers()}
        
    print("\nTraining completed. Store masks for inference.")

# --------------------------
# 6. Run the Experiment
# --------------------------
if __name__ == "__main__":
    train_nse()
