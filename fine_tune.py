import pickle
from resunetPlusPlus_pytorch_copy import build_resunetplusplus

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU available: ", torch.cuda.is_available())

model_path = r'/Users/lauramachlab/Library/CloudStorage/OneDrive-Personal/Documents/_northwestern/_MSAI/c3 lab/resunet_training/OCT-RIPL-Detection/ColonoscopyTrained_resUnetPlusPlus.pkl'    
model = build_resunetplusplus()
model.load_state_dict(torch.load(model_path))
model.to(device)


