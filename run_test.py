from configurations.paths import paths
from data.dataLoader import run_test_
import torch

torch.manual_seed(42)
run_test_()