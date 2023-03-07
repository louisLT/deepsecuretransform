# adapted from https://gist.github.com/koshian2/64e92842bec58749826637e3860f11fa

# # TODO :
# 3) simpler model

import os
import datetime
import logging
import shutil

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

class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu", apply_bn=True):
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
        self.apply_bn = apply_bn
    def forward(self, x):
        return self.activation(self.bn(self.convt(x)) if self.apply_bn else self.convt(x))

class Encoder(nn.Module):
    def __init__(self, color_channels, bottleneck_nb_channels, encoder_output_size, n_latent_features):
        self.bottleneck_nb_channels = bottleneck_nb_channels
        self.n_neurons_in_middle_layer = bottleneck_nb_channels * encoder_output_size
        super().__init__()
        self.bottle = EncoderModule(color_channels, 4, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(4, 8, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(8, 16, stride=2, kernel=3, pad=1)
        self.m3 = EncoderModule(16, self.bottleneck_nb_channels, stride=2, kernel=3, pad=1)
        self.fc = nn.Linear(self.n_neurons_in_middle_layer, n_latent_features)
    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return self.fc(out.view(-1, self.n_neurons_in_middle_layer))

class Decoder(nn.Module):
    def __init__(self, color_channels, encoder_output_size, bottleneck_nb_channels, n_latent_features):
        self.encoder_output_size = encoder_output_size
        self.bottleneck_nb_channels = bottleneck_nb_channels
        self.n_neurons_in_middle_layer = bottleneck_nb_channels * encoder_output_size
        super().__init__()
        self.m1 = DecoderModule(32, 16, stride=1)
        self.m2 = DecoderModule(16, 8, stride=2)
        self.m3 = DecoderModule(8, 4, stride=2)
        self.bottle = DecoderModule(4, color_channels, stride=1, activation="sigmoid")
        self.fc = nn.Linear(n_latent_features, self.n_neurons_in_middle_layer)

    def forward(self, x):
        x = self.fc(x)
        out = x.view(-1, self.bottleneck_nb_channels, 1, self.encoder_output_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)

class VAE(nn.Module):

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = "cpu"

        super().__init__()

        # model parameters
        color_channels = 1
        encoder_output_size = 64
        bottleneck_nb_channels = 32
        n_latent_features = 256

        # Encoder
        self.encoder = Encoder(color_channels,
                               bottleneck_nb_channels,
                               encoder_output_size,
                               n_latent_features)

        # Decoder
        self.decoder = Decoder(color_channels,
                               encoder_output_size,
                               bottleneck_nb_channels,
                               n_latent_features)

        # data
        self.train_loader, self.test_loader = self.load_data()

        # history
        self.history = {
            "train_loss":[],
            "val_loss":[]}

    def forward(self, x):
        z = self.encoder(x)
        d = self.decoder(z)
        return d

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
    def loss_function(self, recon_x, x):
        return F.binary_cross_entropy(recon_x, x, reduction="sum")  # TODO use reduction parameter

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
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
        # save code files
        main_script_path = os.path.realpath(__file__)
        other_filenames = ["generate_time_series.py"]
        files_to_save = [main_script_path] + [
            os.path.join(os.path.dirname(main_script_path), elem) for elem in other_filenames]
        for file_i in files_to_save:
            shutil.copyfile(file_i,
                            os.path.join(self.dump_folder, os.path.basename(file_i)))

    # Train
    def fit_train(self, epoch):
        self.train()
        LOGGER.info(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for batch_idx, inputs in enumerate(self.train_loader):
            inputs = inputs.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()
            recon_batch = self(inputs)
            loss = self.loss_function(recon_batch, inputs)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            samples_cnt += inputs.size(0)

            if batch_idx%50 == 0:
                LOGGER.info(
                    "TRAIN epoch %s, batch %s/%s, loss %s",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    train_loss / samples_cnt)

        self.history["train_loss"].append(train_loss/samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        local_fig_folder = os.path.join(self.dump_folder, str(epoch).zfill(3))
        os.makedirs(local_fig_folder)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.test_loader):
                inputs_ = inputs
                inputs = inputs.to(self.device, dtype=torch.float)
                recon_batch = self(inputs)
                loss = self.loss_function(recon_batch, inputs)

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
                "VAL epoch %s, loss %s",
                epoch,
                val_loss / samples_cnt)

        self.history["val_loss"].append(val_loss/samples_cnt)

    # save results
    def save_history(self):
        file_addr = os.path.join(
            self.dump_folder,
            "history_%s.csv" % str(self.num_version).zfill(3)
        )
        pd.DataFrame(net.history).to_csv(file_addr)

    # def save_model(self, checkpoint_name="model_state"):
    #     file_addr = os.path.join(self.dump_folder, checkpoint_name + ".zip")
    #     LOGGER.info("saving model to %s", file_addr)
    #     torch.save(self.state_dict(), file_addr)

    # def load_model(self, num_version, checkpoint_name="model_state"):
    #     # save on gpu, load on gpu : https://pytorch.org/tutorials/beginner/saving_loading_models.html
    #     dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(num_version).zfill(3))
    #     file_addr = os.path.join(dump_folder, checkpoint_name + ".zip")
    #     LOGGER.info("loading model from %s", file_addr)
    #     self.load_state_dict(torch.load(file_addr))

    def save_model(self, checkpoint_name="model_state"):
        for str_, model_ in [("encoder", self.encoder), ("decoder", self.decoder)]:
            file_addr = os.path.join(self.dump_folder, f"{checkpoint_name}_{str_}.zip")
            LOGGER.info(f"saving {str_} model to {file_addr}")
            torch.save(model_.state_dict(), file_addr)

    def load_model(self, num_version, checkpoint_name="model_state"):
        # save on gpu, load on gpu : https://pytorch.org/tutorials/beginner/saving_loading_models.html
        dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(num_version).zfill(3))
        for str_, model_ in [("encoder", self.encoder), ("decoder", self.decoder)]:
            file_addr = os.path.join(dump_folder, f"{checkpoint_name}_{str_}.zip")
            LOGGER.info(f"loading {str_} model from %s", file_addr)
            model_.load_state_dict(torch.load(file_addr))

if __name__ == "__main__":

    net = VAE()
    nb_epochs = 25
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


