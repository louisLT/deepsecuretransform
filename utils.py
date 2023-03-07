import os
import logging
import shutil

import torch

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)

def get_device(device):
    if device == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert device == "cpu", f"unknown device : {device}"
        return "cpu"

def create_dump_folder(dumps_dir, main_script_path):
    # local dump folder
    existing_folders = sorted(os.listdir(dumps_dir))
    if existing_folders:
        last_version = int(existing_folders[-1][-3:])
    else:
        last_version = -1
    num_version = last_version + 1
    dump_folder = os.path.join(dumps_dir, "version_%s" % str(num_version).zfill(3))
    os.makedirs(dump_folder)
    LOGGER.info("creating new dump folder : %s", dump_folder)
    # save code files
    main_script_path = os.path.realpath(__file__)
    other_filenames = ["time_series.py", "modules.py", "utils.py"]
    files_to_save = [main_script_path] + [
        os.path.join(os.path.dirname(main_script_path), elem) for elem in other_filenames]
    for file_i in files_to_save:
        shutil.copyfile(file_i,
                        os.path.join(dump_folder, os.path.basename(file_i)))
    return num_version, dump_folder

def save_series(list_of_series, local_folder, idx, suffix, ylim):
    """plot list of series on a figure, and in a csv
    """
    # reconstruction
    plt.figure()
    plt.xlim([0, 255])
    if ylim is not None:
        plt.ylim(ylim)
    plt.axis("off")
    df = {}
    for name_i, series_i in list_of_series:
        series_i = pd.Series(series_i)
        df[name_i] = series_i
        series_i.plot(label=name_i)
    df = pd.DataFrame(df)
    file_addr = os.path.join(local_folder, f"{str(idx).zfill(3)}_{suffix}")
    if all(elem[0] is not None for elem in list_of_series):
        plt.legend(loc="upper left", prop={"size": 13})
    plt.savefig(file_addr + ".png")
    plt.close()
    df.to_csv(file_addr + ".csv")