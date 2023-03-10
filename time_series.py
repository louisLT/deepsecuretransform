"""
Author : Louis Le Tarnec
"""

from pandas import Series
from scipy.interpolate import interp1d
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def uniform(min_, max_, rng):
    return rng.uniform(min_, max_) if rng is not None else np.random.uniform(min_, max_)

def create_random_time_series(nb_points, interval_size, rng):
    x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
    y_vals = [uniform(0.2, 0.8, rng) for _ in range(len(x_vals))]
    f = interp1d(x_vals, y_vals, kind='quadratic')
    ts = [float(f(i)) for i in range(nb_points)]
    min_ = min(ts)
    rand_ = uniform(0, 0.48, rng)
    ts = [elem - min_ + rand_ + uniform(0, 0.02, rng) for elem in ts]
    ts = [max(min(1, i), 0) for i in ts]
    return np.array(ts)[None, None, :]

class TimeSeriesDataset(Dataset):
    """
    single random realistic time series generator
    """

    def __init__(self, nb_points, interval_size, nb_samples, seed=None):
        super(TimeSeriesDataset).__init__()
        self.nb_points = nb_points
        self.interval_size = interval_size
        self.nb_samples = nb_samples
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        return create_random_time_series(self.nb_points, self.interval_size, self.rng)

class SumNTimeSeriesDataset(Dataset):
    """
    generat n random realistic time series and the corresponding sum
    """

    def __init__(self, nb_points, interval_size, nb_samples, n_series, seed=None):
        super(SumNTimeSeriesDataset).__init__()
        self.nb_points = nb_points
        self.interval_size = interval_size
        self.nb_samples = nb_samples
        self.n_series = n_series
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        series = []
        for _ in range(self.n_series):
            series.append(create_random_time_series(self.nb_points, self.interval_size, self.rng))
        mean_ = np.mean(series, axis=0)
        return tuple(series) + (mean_,)

if __name__ == "__main__":

    import time

    test = "one_time_series"

    if test == "one_time_series":
        # test TimeSeriesDataset


        nb_samples = 10
        num_workers = 0
        seed = None  if num_workers > 0 else 103
        plot = True

        ds = TimeSeriesDataset(nb_points=256, interval_size=24, nb_samples=nb_samples, seed=seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=num_workers)

        tps_1 = time.time()
        plt.figure()
        plt.xlim([0, 255])
        plt.ylim([0, 1.25])
        plt.axis("off")
        for idx_i, series_i in enumerate(dl):
            if plot:
                series_ = Series(series_i[0, 0, 0])
                series_.plot(label=str(idx_i))
        tps_2 = time.time()
        print("elapsed time : ", tps_2 - tps_1)

    else:
        # test SumNTimeSeriesDataset

        nb_samples = 6
        num_workers = 0
        seed = None  if num_workers > 0 else 103
        plot = True
        n_series = 2

        ds = SumNTimeSeriesDataset(nb_points=256,
                                interval_size=24,
                                nb_samples=nb_samples,
                                n_series=n_series,
                                seed=seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=num_workers)

        tps_1 = time.time()
        for idx_i, series in enumerate(dl):
            if plot:
                plt.figure()
                for ts_ in series:
                    series_ = Series(ts_[0, 0, 0])
                    series_.plot()
        tps_2 = time.time()
        print("elapsed time : ", tps_2 - tps_1)