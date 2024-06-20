import pynvml
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Function to find the GPU with the most free memory
def get_best_gpu():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    best_gpu = None
    max_free_memory = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i
    pynvml.nvmlShutdown()
    return best_gpu

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GCNConv):
            nn.init.kaiming_normal_(m.lin.weight, nonlinearity='relu')
            if m.lin.bias is not None:
                nn.init.constant_(m.lin.bias, 0)
