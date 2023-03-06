
import random

from pandas import Series
from scipy.interpolate import interp1d
import numpy as np
import torch
from torch.utils.data import Dataset

# test llt

# def create_random_time_series(nb_points, interval_size): # 63
#     x_vals = list(range(0, nb_points, interval_size)) + [nb_points]

#     # y_vals = [random.uniform(0, 1) for _ in range(len(x_vals))]
#     # test llt
#     y_vals = [random.uniform(0.2, 0.8) for _ in range(len(x_vals))]
#     # y_vals = [random.uniform(-1, 1) for _ in range(len(x_vals))]

#     f = interp1d(x_vals, y_vals, kind='quadratic')
#     ts = [float(f(i)) for i in range(nb_points)]

#     # test llt
#     ts = [max(min(1, i), 0) for i in ts]

#     # test llt
#     return (np.array(ts) - np.min(ts) + random.uniform(0, 0.5))[None, None, :]
#     # return np.array(ts)[None, None, :]

# def create_random_time_series(nb_points, interval_size):
#     x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
#     y_vals = [random.uniform(0.3, 0.7) for _ in range(len(x_vals))]
#     f = interp1d(x_vals, y_vals, kind='quadratic')
#     ts = [float(f(i)) for i in range(nb_points)]
#     ts = [max(min(1, i), 0) for i in ts]
#     return np.array(ts)[None, None, :]

# def create_random_time_series(nb_points, interval_size): # 74
#     x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
#     y_vals = [random.uniform(0.2, 0.7) for _ in range(len(x_vals))]
#     f = interp1d(x_vals, y_vals, kind='quadratic')
#     ts = [float(f(i)) for i in range(nb_points)]
#     min_ = min(ts)
#     rand_ = random.uniform(0, 0.5)
#     ts = [elem - min_ + rand_ for elem in ts]
#     ts = [max(min(1, i), 0) for i in ts]
#     return np.array(ts)[None, None, :]

# def create_random_time_series(nb_points, interval_size): # 75
#     x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
#     y_vals = [random.uniform(0.2, 0.7) for _ in range(len(x_vals))]
#     f = interp1d(x_vals, y_vals, kind='quadratic')
#     ts = [float(f(i)) for i in range(nb_points)]
#     min_ = min(ts)
#     rand_ = random.uniform(0, 0.4)
#     ts = [elem - min_ + rand_ + random.uniform(0, 0.1) for elem in ts]
#     ts = [max(min(1, i), 0) for i in ts]
#     return np.array(ts)[None, None, :]

# def create_random_time_series(nb_points, interval_size): # 76
#     x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
#     y_vals = [random.uniform(0.2, 0.8) for _ in range(len(x_vals))]
#     f = interp1d(x_vals, y_vals, kind='quadratic')
#     ts = [float(f(i)) for i in range(nb_points)]
#     min_ = min(ts)
#     rand_ = random.uniform(0, 0.4)
#     ts = [elem - min_ + rand_ + random.uniform(0, 0.1) for elem in ts]
#     ts = [max(min(1, i), 0) for i in ts]
#     return np.array(ts)[None, None, :]

def create_random_time_series(nb_points, interval_size): # 77
    x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
    y_vals = [random.uniform(0.2, 0.8) for _ in range(len(x_vals))]
    f = interp1d(x_vals, y_vals, kind='quadratic')
    ts = [float(f(i)) for i in range(nb_points)]
    min_ = min(ts)
    rand_ = random.uniform(0, 0.5)
    ts = [elem - min_ + rand_ for elem in ts]
    ts = [max(min(1, i), 0) for i in ts]
    return np.array(ts)[None, None, :]

class TimeSeriesDataset(Dataset):

    def __init__(self, nb_points, interval_size, nb_samples):
        super(TimeSeriesDataset).__init__()
        self.nb_points = nb_points
        self.interval_size = interval_size
        self.nb_samples = nb_samples

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        return create_random_time_series(self.nb_points, self.interval_size)

if __name__ == "__main__":

    import time

    nb_samples = 6
    num_workers = 0
    plot = True

    ds = TimeSeriesDataset(nb_points=256, interval_size=24, nb_samples=nb_samples)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=num_workers)

    tps_1 = time.time()
    for idx_i, series_i in enumerate(dl):
        if plot:
            series_ = Series(series_i[0, 0, 0])
            series_.plot()
    tps_2 = time.time()
    print("elapsed time : ", tps_2 - tps_1)



    # plt.figure()
    # tps_1 = time.time()
    # for idx_i, series_i in enumerate(dl):
    #     if plot:
    #         series_ = Series(series_i[0, 0, 0])
    #         series_.plot()
    # tps_2 = time.time()
    # print("elapsed time : ", tps_2 - tps_1)
    # import matplotlib.pyplot as plt
    # plt.savefig('/tmp/figure2.png')
