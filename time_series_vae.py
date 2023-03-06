# adapted from https://gist.github.com/koshian2/64e92842bec58749826637e3860f11fa

import os
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt

import generate_time_series

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s',
                    level=logging.INFO,
                    force=True)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
DUMPS_DIR = os.environ["DUMPS_DIR"]

class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 4, stride=1, kernel=1, pad=0)

        self.m1 = EncoderModule(4, 8, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(8, 16, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(16, 32, stride=pooling_kernels[1], kernel=3, pad=1)
        # test llt
        # self.m1 = EncoderModule(4, 4, stride=1, kernel=3, pad=1)
        # self.m2 = EncoderModule(4, 4, stride=pooling_kernels[0], kernel=3, pad=1)
        # self.m3 = EncoderModule(4, 4, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)

class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels,
                                        output_channels,
                                        kernel_size=(1, stride),
                                        stride=(1, stride))
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            assert False, "unknown activation : %s" % activation

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))

class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()

        self.m1 = DecoderModule(32, 16, stride=1)
        self.m2 = DecoderModule(16, 8, stride=pooling_kernels[1])
        self.m3 = DecoderModule(8, 4, stride=pooling_kernels[0])
        # test llt
        # self.m1 = DecoderModule(4, 4, stride=1)
        # self.m2 = DecoderModule(4, 4, stride=pooling_kernels[1])
        # self.m3 = DecoderModule(4, 4, stride=pooling_kernels[0])

        # test llt
        # self.bottle = DecoderModule(4, color_channels, stride=1, activation="relu")
        self.bottle = DecoderModule(4, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):

        out = x.view(-1, 32, 1, self.decoder_input_size)
        # test llt
        # out = x.view(-1, 4, 1, self.decoder_input_size)

        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)

class VAE(nn.Module):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # test llt
        # self.device = "cpu"

        super().__init__()
        # latent features

        # self.n_latent_features = 64
        # test llt
        self.n_latent_features = 256

        # resolution
        pooling_kernel = [2, 2]
        encoder_output_size = 64

        # color channels
        color_channels = 1

        # kld loss factor

        # self.kld_loss_factor = 0.05
        # test llt
        self.kld_loss_factor = 0

        # neurons int middle layer

        n_neurons_middle_layer = 32 * encoder_output_size
        # test llt
        # n_neurons_middle_layer = 4 * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        # data
        self.train_loader, self.test_loader = self.load_data()
        # history
        self.history = {
            "train_bce_loss": [],
            "train_kld_loss": [],
            "train_loss":[],
            "val_bce_loss": [],
            "val_kld_loss": [],
            "val_loss":[]}

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = 0  # test llt torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc3(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar

    # Data
    def load_data(self):

        ds = generate_time_series.TimeSeriesDataset(nb_points=256,
                                                    interval_size=24,
                                                    nb_samples=1024)
        dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4)

        train_loader = dl
        test_loader = dl

        return train_loader, test_loader

    # Model
    def loss_function(self, recon_x, x, mu, logvar):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        # bce = F.mse_loss(recon_x, x)
        # test llt
        bce = F.binary_cross_entropy(recon_x, x, size_average=False)

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce, kld

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)  # test llt 1e-3
        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    def init_dump_folder(self):
        # local dump folder
        existing_folders = sorted(os.listdir(DUMPS_DIR))
        if existing_folders:
            last_version = int(existing_folders[-1][-2:])
        else:
            last_version = -1
        self.num_version = last_version + 1
        self.dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(self.num_version).zfill(3))
        os.makedirs(self.dump_folder)
        LOGGER.info("creating new dump folder : %s", self.dump_folder)

    # Train
    def fit_train(self, epoch):
        self.train()
        LOGGER.info(f"Epoch: {epoch:d} {datetime.datetime.now()}")
        bce_loss = 0
        kld_loss = 0
        train_loss = 0
        samples_cnt = 0
        for batch_idx, inputs in enumerate(self.train_loader):
            inputs = inputs.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(inputs)

            bce, kld = self.loss_function(recon_batch, inputs, mu, logvar)
            loss = bce + self.kld_loss_factor * kld
            loss.backward()
            self.optimizer.step()

            bce_loss += bce.item()
            kld_loss += kld.item()
            train_loss += loss.item()

            samples_cnt += inputs.size(0)

            if batch_idx%50 == 0:
                LOGGER.info(
                    "TRAIN epoch %s, batch %s/%s, bce loss %s kld loss %s loss %s",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    bce_loss / samples_cnt,
                    kld_loss / samples_cnt,
                    train_loss / samples_cnt)

        self.history["train_bce_loss"].append(bce_loss/samples_cnt)
        self.history["train_kld_loss"].append(kld_loss/samples_cnt)
        self.history["train_loss"].append(train_loss/samples_cnt)

    def test(self, epoch):
        self.eval()
        bce_loss = 0
        kld_loss = 0
        val_loss = 0
        samples_cnt = 0
        local_fig_folder = os.path.join(self.dump_folder, str(epoch).zfill(3))
        os.makedirs(local_fig_folder)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.test_loader):
                inputs_ = inputs
                inputs = inputs.to(self.device, dtype=torch.float)
                recon_batch, mu, logvar = self(inputs)
                bce, kld = self.loss_function(recon_batch, inputs, mu, logvar)
                loss = bce + self.kld_loss_factor * kld

                bce_loss += bce.item()
                kld_loss += kld.item()
                val_loss += loss.item()

                samples_cnt += inputs.size(0)

                if batch_idx == 0:
                    for i_fig in range(min(inputs.size(0), 20)):
                        plt.figure()
                        series_input = pd.Series(inputs_[i_fig, 0, 0])
                        series_input.plot()
                        series_output = pd.Series(recon_batch[i_fig, 0, 0].cpu())
                        series_output.plot()
                        file_addr = os.path.join(local_fig_folder,
                                                 str(i_fig).zfill(3) + ".png")
                        plt.savefig(file_addr)
                        plt.close()


            LOGGER.info(
                "VAL epoch %s, bce loss %s kld loss %s loss %s",
                epoch,
                bce_loss / samples_cnt,
                kld_loss / samples_cnt,
                val_loss / samples_cnt)

        self.history["val_bce_loss"].append(bce_loss/samples_cnt)
        self.history["val_kld_loss"].append(kld_loss/samples_cnt)
        self.history["val_loss"].append(val_loss/samples_cnt)

        # # sampling
        # file_addr = os.path.join(
        #     self.dump_folder,
        #     f"epoch_{str(epoch)}_sampling.png"
        # )
        # save_image(self.sampling(), file_addr, nrow=8)

    # save results
    def save_history(self):
        file_addr = os.path.join(
            self.dump_folder,
            "history.csv"
        )
        pd.DataFrame(net.history).to_csv(file_addr)

    def save_model(self, checkpoint_name="model_state"):
        file_addr = os.path.join(self.dump_folder, checkpoint_name + ".zip")
        LOGGER.info("saving model to %s", file_addr)
        torch.save(self.state_dict(), file_addr)

    def load_model(self, num_version, checkpoint_name="model_state"):
        # save on gpu, load on gpu : https://pytorch.org/tutorials/beginner/saving_loading_models.html
        dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(num_version).zfill(3))
        file_addr = os.path.join(dump_folder, checkpoint_name + ".zip")
        LOGGER.info("loading model from %s", file_addr)
        self.load_state_dict(torch.load(file_addr))

if __name__ == "__main__":

    net = VAE()
    nb_epochs = 300 # test llt 50
    net.init_model()
    net.init_dump_folder()
    for i in range(nb_epochs):
        net.fit_train(i + 1)
        net.test(i + 1)
    net.save_history()
    net.save_model()

    # load
    # net_2 = VAE()
    # net_2.load_model(net.num_version)
    # net_2.init_model()
    # net_2.init_dump_folder()
    # net_2.test(9999)


