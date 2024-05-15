import numpy as np
import random
import torch

def set_seeds(args, rank=0):
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + rank)
