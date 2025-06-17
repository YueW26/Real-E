import os
import fire
import numpy as np
import pandas as pd


def gen_stamp(data_path,data_root,cycle):
    data = pd.read_csv(data_path, header=None).values.astype(float)
    T, N = data.shape
    time_stamp = np.zeros(T)
    for idx in range(T):
        time_stamp[idx] = idx % cycle
    root = data_root
    name = "time_stamp_F_96_R_0301.npy" ## EnergyTSF/TPGNN/datasets/time_stamp.npy
    print(os.path.join(root, name))
    np.save(os.path.join(root, name), time_stamp)

if __name__ == '__main__':
    fire.Fire()