"""
Fast NMNIST → PyTorch tensor converter.
Single pass: reads binary event files, applies ToFrame, stacks into .pt tensors.
Result: instant loading for training (vs 90 min DiskCachedDataset).

Run once: python examples/prep_nmnist.py
Output: ./data/nmnist_train.pt (~5.5GB), ./data/nmnist_test.pt (~0.9GB)
"""
import sys, os, time
import numpy as np
import torch
import tonic
import tonic.transforms as ttf

# Override DiskCachedDataset in demo_a_perception before import
# (We're standalone — no dependency on demo_a)

SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
N_TIME_BINS = 10

frame_tf = ttf.Compose([
    ttf.Denoise(filter_time=10000),
    ttf.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=N_TIME_BINS),
])

for split, name in [(True, "train"), (False, "test")]:
    fname = f"./data/nmnist_{name}.pt"
    if os.path.exists(fname):
        print(f"{fname} exists, skipping ({os.path.getsize(fname)/1024/1024:.0f} MB)")
        continue

    print(f"\nLoading N-MNIST {name} set...")
    dataset = tonic.datasets.NMNIST(save_to="./data", train=split, transform=frame_tf)
    n = len(dataset)
    print(f"  {n} samples — converting to single .pt tensor...")

    frames_list = []
    targets_list = []
    t0 = time.time()
    for i in range(n):
        frames, target = dataset[i]
        frames_list.append(torch.from_numpy(frames).float())
        targets_list.append(target)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  {i+1}/{n} ({100*(i+1)/n:.0f}%) — {rate:.0f} samples/sec, ETA {eta/60:.1f} min")

    data = torch.stack(frames_list)
    targets = torch.tensor(targets_list)
    elapsed = time.time() - t0
    print(f"  Done: {data.shape}, {targets.shape} in {elapsed/60:.1f} min")

    print(f"  Saving {fname} ({data.element_size() * data.numel() / 1024/1024:.0f} MB)...")
    torch.save({"data": data, "targets": targets}, fname)
    print(f"  Saved: {os.path.getsize(fname)/1024/1024:.0f} MB")

print("\nDone! Train + test tensors ready.")
print("Delete ./cache/nmnist/ to free space if needed.")
