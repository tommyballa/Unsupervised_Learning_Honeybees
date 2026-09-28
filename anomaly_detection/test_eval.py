import sys
sys.path.append('src')
import torch
from moviad.models.fastflow.fastflow import AnomalyMapGenerator
hidden_variables = [torch.randn(10, 256, 64, 64) * 15, torch.randn(10, 512, 32, 32) * 15, torch.randn(10, 1024, 16, 16) * 15]
gen = AnomalyMapGenerator(input_size=(256, 256))
anomaly_maps = gen(hidden_variables)
anomaly_scores = anomaly_maps.view(10, -1).max(dim=1).values
print(anomaly_scores)
