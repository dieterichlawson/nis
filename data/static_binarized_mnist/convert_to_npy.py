import os
import numpy as np

for filename in ["train.amat", "test.amat", "valid.amat"]:
  x = np.loadtxt(filename, delimiter=" ", dtype=np.uint8)
  base =  os.path.splitext(filename)[0]
  np.save(base, x)
  if filename == "train.amat":
    np.save("train_mean", np.mean(x, axis=0))
