# inspired from https://gist.github.com/koshian2/64e92842bec58749826637e3860f11fa

import os
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

import time_series
import modules
import utils

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s',
                    level=logging.INFO,
                    force=True)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
DUMPS_DIR = os.environ["DUMPS_DIR"]
COLOR_CHANNELS = 1
ENCODER_OUTPUT_SIZE = 64
BOTTLENECK_NB_CHANNELS = 32
N_LATENT_FEATURES = 256


class Autoencoder(nn.Module):

    def __init__(self, device="gpu"):

        self.device = utils.get_device(device)

        super().__init__()

        # encoder
        self.encoder = modules.Encoder(COLOR_CHANNELS,
                                       BOTTLENECK_NB_CHANNELS,
                                       ENCODER_OUTPUT_SIZE,
                                       N_LATENT_FEATURES)

        # decoder
        self.decoder = modules.Decoder(COLOR_CHANNELS,
                                       BOTTLENECK_NB_CHANNELS,
                                       ENCODER_OUTPUT_SIZE,
                                       N_LATENT_FEATURES)

        # data
        self.train_loader = self.load_data(seed=None, num_workers=4, nb_samples=1024, batch_size=64)

        # history
        self.history = {"train_loss":[], "val_loss":[]}

    def forward(self, x):
        if self.part == "decoder":
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)
        d = self.decoder(z)
        return d

    def load_data(self, seed, num_workers, nb_samples, batch_size):
        ds = time_series.TimeSeriesDataset(nb_points=256,
                                           interval_size=24,
                                           nb_samples=nb_samples,
                                           seed=seed)
        return torch.utils.data.DataLoader(ds,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)

    def loss_function(self, recon_x, x):
        return F.binary_cross_entropy(recon_x, x, reduction="sum")

    def init_model(self, part="*"):
        self.part = part
        if part == "*":
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        elif part == "encoder":
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)
        else:
            assert part == "decoder", f"unknown part : {part}"
            self.optimizer = optim.Adam(self.decoder.parameters(), lr=1e-3)
        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    def init_dump_folder(self):
        main_script_path = os.path.realpath(__file__)
        self.num_version, self.dump_folder = utils.create_dump_folder(DUMPS_DIR, main_script_path)

    def fit_train(self, epoch):
        self.train()
        LOGGER.info(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for inputs in self.train_loader:
            inputs = inputs.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()
            recon_batch = self(inputs)
            loss = self.loss_function(recon_batch, inputs)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            samples_cnt += inputs.size(0)
        LOGGER.info("TRAIN epoch %s, loss %s", epoch, train_loss / samples_cnt)
        self.history["train_loss"].append(train_loss/samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        local_folder = os.path.join(self.dump_folder, str(epoch).zfill(3))
        os.makedirs(local_folder)
        test_loader = self.load_data(seed=103, num_workers=0, nb_samples=64, batch_size=64)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                inputs_ = inputs
                inputs = inputs.to(self.device, dtype=torch.float)
                z = self.encoder(inputs)
                recon_batch = self.decoder(z)
                loss = self.loss_function(recon_batch, inputs)
                val_loss += loss.item()
                samples_cnt += inputs.size(0)
                max_nb_figures = 20
                if batch_idx == 0:
                    for i_fig in range(min(inputs.size(0), max_nb_figures)):
                        # reconstruction
                        utils.save_series(
                            [("input", inputs_[i_fig, 0, 0].cpu()), ("reconstruction", recon_batch[i_fig, 0, 0].cpu())],
                            local_folder, i_fig, "recon")
                        # encoding
                        utils.save_series(
                            [("encoded", z[i_fig].cpu())],
                            local_folder, i_fig, "encode")
        LOGGER.info("VAL epoch %s, loss %s", epoch, val_loss / samples_cnt)
        self.history["val_loss"].append(val_loss/samples_cnt)

    def save_history(self):
        file_addr = os.path.join(
            self.dump_folder,
            f"history_{str(self.num_version).zfill(3)}.csv"
        )
        pd.DataFrame(self.history).to_csv(file_addr)

    def save_model(self, checkpoint_name="model_state"):
        for str_, model_ in [("encoder", self.encoder), ("decoder", self.decoder)]:
            file_addr = os.path.join(self.dump_folder, f"{checkpoint_name}_{str_}.zip")
            LOGGER.info(f"saving {str_} model to {file_addr}")
            torch.save(model_.state_dict(), file_addr)

    def load_model(self, num_version, checkpoint_name="model_state", part="*"):
        # save on gpu, load on gpu : https://pytorch.org/tutorials/beginner/saving_loading_models.html
        assert part in ["encoder", "decoder", "*"], f"unknown part : {part}"
        dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(num_version).zfill(3))
        for str_, model_ in [("encoder", self.encoder), ("decoder", self.decoder)]:
            if part in [str_, "*"]:
                file_addr = os.path.join(dump_folder, f"{checkpoint_name}_{str_}.zip")
                LOGGER.info(f"loading {str_} model from %s", file_addr)
                model_.load_state_dict(torch.load(file_addr))


class SumDecoder(nn.Module):

    def __init__(self, n_time_series=3, device="gpu"):

        self.n_time_series = n_time_series
        self.device = utils.get_device(device)

        super().__init__()

        # define three different encoders
        self.encoders = []
        for _ in range(n_time_series):
            self.encoders.append(modules.Encoder(COLOR_CHANNELS,
                                                 BOTTLENECK_NB_CHANNELS,
                                                 ENCODER_OUTPUT_SIZE,
                                                 N_LATENT_FEATURES))

        # decoder
        self.decoder = modules.Decoder(COLOR_CHANNELS,
                                       BOTTLENECK_NB_CHANNELS,
                                       ENCODER_OUTPUT_SIZE,
                                       N_LATENT_FEATURES * n_time_series)

        # data
        self.train_loader = self.load_data(seed=None,
                                           num_workers=4,
                                           nb_samples=1024,
                                           batch_size=64)

        # history
        self.history = {"train_loss":[], "val_loss":[]}

    def forward(self, *args):
        assert len(args) == self.n_time_series, f"got {len(args)} args, expected {self.n_time_series}"
        with torch.no_grad():
            encoded = []
            for idx in range(self.n_time_series):
                encoded.append(self.encoders[idx](args[idx]))
            concat_ = torch.cat(encoded, dim=1)
        decoded = self.decoder(concat_)
        return decoded

    def load_data(self, seed, num_workers, nb_samples, batch_size):
        ds = time_series.SumNTimeSeriesDataset(nb_points=256,
                                               interval_size=24,
                                               nb_samples=nb_samples,
                                               n_series=self.n_time_series,
                                               seed=seed)
        return torch.utils.data.DataLoader(ds,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)

    def loss_function(self, recon_x, x):
        return F.binary_cross_entropy(recon_x, x, reduction="sum")

    def init_model(self):
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=1e-3)
        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    def init_dump_folder(self):
        main_script_path = os.path.realpath(__file__)
        self.num_version, self.dump_folder = utils.create_dump_folder(DUMPS_DIR, main_script_path)

    def fit_train(self, epoch):
        self.train()
        LOGGER.info(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for sample_ in self.train_loader:
            assert len(sample_) == self.n_time_series + 1, f"got {len(sample_)} args, expected {self.n_time_series}"
            sample_ = [elem.to(self.device, dtype=torch.float) for elem in sample_]
            inputs = sample_[:self.n_time_series]
            target = sample_[-1]
            self.optimizer.zero_grad()
            recon_batch = self(inputs)
            loss = self.loss_function(recon_batch, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            samples_cnt += inputs.size(0)
        LOGGER.info("TRAIN epoch %s, loss %s", epoch, train_loss / samples_cnt)
        self.history["train_loss"].append(train_loss/samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        local_folder = os.path.join(self.dump_folder, str(epoch).zfill(3))
        os.makedirs(local_folder)
        test_loader = self.load_data(seed=103, num_workers=0, nb_samples=64, batch_size=64)
        with torch.no_grad():
            for batch_idx, sample_ in enumerate(test_loader):
                assert len(sample_) == self.n_time_series + 1, f"got {len(sample_)} args, expected {self.n_time_series}"
                sample_ = [elem.to(self.device, dtype=torch.float) for elem in sample_]
                inputs = sample_[:self.n_time_series]
                target = sample_[-1]
                recon_batch = self(inputs)
                loss = self.loss_function(recon_batch, inputs)
                val_loss += loss.item()
                samples_cnt += inputs.size(0)
                max_nb_figures = 20
                if batch_idx == 0:
                    for i_fig in range(min(inputs.size(0), max_nb_figures)):
                        utils.save_series(
                            [("target", target[i_fig, 0, 0].cpu()), ("reconstruction", recon_batch[i_fig, 0, 0].cpu())],
                            local_folder, i_fig, "recon")
        LOGGER.info("VAL epoch %s, loss %s", epoch, val_loss / samples_cnt)
        self.history["val_loss"].append(val_loss/samples_cnt)

    def save_history(self):
        file_addr = os.path.join(
            self.dump_folder,
            f"history_{str(self.num_version).zfill(3)}.csv"
        )
        pd.DataFrame(self.history).to_csv(file_addr)

    def save_model(self, checkpoint_name="model_state"):
        """
        save only decoder, only trained model
        """
        file_addr = os.path.join(self.dump_folder, f"{checkpoint_name}_decoder.zip")
        LOGGER.info(f"saving decoder model to {file_addr}")
        torch.save(self.decoder.state_dict(), file_addr)

    def load_models(self, nums_version, checkpoint_name="model_state"):
        """
        load encoders
        """
        assert len(nums_version) == self.n_time_series, f"got {len(nums_version)} args, expected {self.n_time_series}"
        for idx_i, num_version_i in enumerate(nums_version):
            dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(num_version_i).zfill(3))
            file_addr = os.path.join(dump_folder, f"{checkpoint_name}_encoder.zip")
            LOGGER.info(f"loading encoder model from %s", file_addr)
            self.encoders[idx_i].load_state_dict(torch.load(file_addr))


if __name__ == "__main__":

    pass

    net = Autoencoder(device="gpu")
    nb_epochs = 25
    net.init_model()
    net.init_dump_folder()
    for i in range(nb_epochs):
        net.fit_train(i + 1)
        net.test(i + 1)
    net.save_history()
    net.save_model()

    # load
    # net_2 = Autoencoder(device="gpu")
    # net_2.load_model(net.num_version)
    # net_2.init_model()
    # net_2.init_dump_folder()
    # net_2.test(9999)

    # train only decoder
    # net = Autoencoder(device="gpu")
    # nb_epochs = 25
    # # net.load_model(106, part="encoder")
    # net.init_model(part="decoder")
    # net.init_dump_folder()
    # for i in range(nb_epochs):
    #     net.fit_train(i + 1)
    #     net.test(i + 1)
    # net.save_history()
    # net.save_model()

    # train sum
    net = SumDecoder(n_time_series=2, device="gpu")
    nb_epochs = 25
    net.load_models([124, 125])
    net.init_model()
    net.init_dump_folder()
    for i in range(nb_epochs):
        net.fit_train(i + 1)
        net.test(i + 1)
    net.save_history()
    net.save_model()
