import os
import time
import math
import pickle
from contextlib import nullcontext
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm,trange

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

batch_size = 32
interval = 64
dimensions =  [
            'cl_op_t',
            'hi_op_t',
            'lo_op_t',
            'op_cl_t_1',
            'Volume',
            'Day',
            'Month',
            'Weekday',
        ]
ind_dim = 8
n_embd = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch():
    
    stocks = []
    input_dir = 'dataset/data'
    for file in os.listdir(input_dir):
        stocks.append(file)
    stock = random.choice(stocks)
    retrived = False
    while not retrived:
        df = pd.read_csv(os.path.join(input_dir,stock))
        df.dropna(axis=0,inplace=True)
        if len(df) > interval + batch_size + 1:
            i = np.random.randint(0, len(df) - batch_size - 1-interval, 1)
            x = torch.stack([torch.from_numpy((df[i:i+interval])[dimensions].to_numpy(dtype=np.float32)) for i in range(batch_size)])
            y = torch.stack([
                    torch.tensor(df.iloc[i+j+1+interval][dimensions].values[0][:1], dtype=torch.float32)
                for j in range(batch_size)
            ])
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            retrived = True
        else:
            os.remove(os.path.join(input_dir,stock))
            stock = random.choice(stocks)
    return x,y



config = GPTConfig()
model = GPT(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


losses = []
for steps in range(100):
    lossi = 0
    for i in range(10):
        x,y = get_batch()
        logits, loss = model(x,y)    
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss.item())
        lossi+=loss.item()
    losses.append(lossi/1000)
    
plt.plot(losses)
plt.show()