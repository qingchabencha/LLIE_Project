import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

model = nn.Conv2d(3, 3, kernel_size=3, padding=1)  # Example model

initial_lr = 1e-3
min_lr = 1e-6
gamma = 0.988

def lr_lambda(epoch):
    return max(gamma ** epoch, min_lr / initial_lr)

optimizer = AdamW(model.parameters(), lr=initial_lr)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(400):  # Example training loop
    # Simulate training step
    optimizer.zero_grad()
    # loss.backward()  # Compute gradients (not shown here)
    optimizer.step()
    
    # Update learning rate
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")