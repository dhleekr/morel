import torch
import torch.nn as nn

a = torch.tensor([5.483363628387451], dtype=torch.float)

torch.save(a, './models/threshold_hopper-expert-v2.pt')