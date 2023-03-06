
from random import gauss

from pandas import Series
from scipy.interpolate import interp1d
import numpy as np

def create_random_time_series(nb_points=256, interval_size=24):
    x_vals = list(range(0, nb_points, interval_size)) + [nb_points]
    y_vals = [gauss(0.0, 1.0) for _ in range(len(x_vals))]
    f = interp1d(x_vals, y_vals, kind='quadratic')
    series = np.array([float(f(i)) for i in range(nb_points)])
    return series

if __name__ == "__main__":

    series = create_random_time_series()

    series_ = Series(series)
    series_.plot()

    # convert to pytorch



