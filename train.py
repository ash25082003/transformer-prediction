import os
import random
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch

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
    input_dir = 'dataset/data'
    stocks = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not stocks:
        raise RuntimeError(f"No CSV files in {input_dir}")

    tried = set()
    while True:
        remaining = [s for s in stocks if s not in tried]
        if not remaining:
            raise RuntimeError(
                f"No file in {input_dir} has >= {interval + batch_size + 1} rows after dropna"
            )
        stock = random.choice(remaining)
        tried.add(stock)

        df = pd.read_csv(os.path.join(input_dir, stock))
        df.dropna(axis=0, inplace=True)
        df = df.reset_index(drop=True)
        if len(df) <= interval + batch_size + 1:
            continue

        start = int(np.random.randint(0, len(df) - batch_size - 1 - interval))
        feats = df[dimensions].to_numpy(dtype=np.float32)
        target_col = dimensions.index('cl_op_t')

        x = torch.stack([
            torch.from_numpy(feats[start + j : start + j + interval])
            for j in range(batch_size)
        ])
        y = torch.tensor(
            [feats[start + j + interval, target_col] for j in range(batch_size)],
            dtype=torch.float32,
        ).unsqueeze(-1)

        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        return x, y



config = GPTConfig()
model = GPT(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


inner_iters = 10
outer_iters = 100
losses = []
for step in range(outer_iters):
    lossi = 0.0
    for _ in range(inner_iters):
        x, y = get_batch()
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss.item())
        lossi += loss.item()
    losses.append(lossi / inner_iters)

plt.plot(losses)
plt.show()