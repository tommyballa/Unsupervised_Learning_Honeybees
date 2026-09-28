import sys
sys.path.append('src')
import torch
from moviad.models.fastflow.fastflow import create_fastflow, FastFlowTrainArgs
from torchvision import transforms

model = create_fastflow((256, 256), "wide_resnet50_2", "cuda:0")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
train_args = FastFlowTrainArgs(epochs=1, batch_size=2)
train_args.init_train(model)

batch = torch.randn(2, 3, 256, 256).cuda()

for i in range(10):
    hidden_variables, log_jacobians = model(batch)
    loss = train_args.loss_function(hidden_variables, log_jacobians)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {i} Loss: {loss.item()}")
