
import random

from pandas import Series
from scipy.interpolate import interp1d
import numpy as np
import torch
from torch.utils.data import Dataset

def create_random_time_series(nb_points, interval_size):
    x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
    y_vals = [random.uniform(0, 1) for _ in range(len(x_vals))]
    f = interp1d(x_vals, y_vals, kind='quadratic')
    ts = [float(f(i)) for i in range(nb_points)]
    return (np.array(ts) - np.min(ts) + random.uniform(0, 0.5))[None, None, :]

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

    nb_samples = 5
    num_workers = 0
    plot = True

    ds = TimeSeriesDataset(nb_points=256, interval_size=24, nb_samples=nb_samples)
    dl = torch.utils.data.DataLoader(ds, batch_size=3, num_workers=num_workers)

    tps_1 = time.time()
    for idx_i, series_i in enumerate(dl):
        if plot:
            series_ = Series(series_i[0, 0, 0])
            series_.plot()
    tps_2 = time.time()
    print("elapsed time : ", tps_2 - tps_1)

