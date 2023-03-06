# adapted from https://gist.github.com/koshian2/64e92842bec58749826637e3860f11fa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import pandas as pd
import os
import datetime
import logging

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
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)

class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))

class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)

class VAE(nn.Module):
    def __init__(self, dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert dataset in ["mnist" ,"fashion-mnist", "cifar", "stl"]

        super().__init__()
        # # latent features
        self.n_latent_features = 64

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            color_channels = 1
        else:
            color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        # data
        self.train_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss":[], "val_loss":[]}

        # local dump folder
        existing_folders = sorted(os.listdir(DUMPS_DIR))
        if existing_folders:
            last_version = int(existing_folders[-1][-2:])
        else:
            last_version = -1
        self.dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(last_version + 1).zfill(3))
        os.makedirs(self.dump_folder)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
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
    def load_data(self, dataset):
        data_transform = transforms.Compose([
                transforms.ToTensor()
        ])
        if dataset == "mnist":
            train = MNIST(root=PATH_DATASETS, train=True, transform=data_transform, download=True)
            test = MNIST(root=PATH_DATASETS, train=False, transform=data_transform, download=True)
        elif dataset == "fashion-mnist":
            train = FashionMNIST(root=PATH_DATASETS, train=True, transform=data_transform, download=True)
            test = FashionMNIST(root=PATH_DATASETS, train=False, transform=data_transform, download=True)
        elif dataset == "cifar":
            train = CIFAR10(root=PATH_DATASETS, train=True, transform=data_transform, download=True)
            test = CIFAR10(root=PATH_DATASETS, train=False, transform=data_transform, download=True)
        elif dataset == "stl":
            train = STL10(root=PATH_DATASETS, split="unlabeled", transform=data_transform, download=True)
            test = STL10(root=PATH_DATASETS, split="test", transform=data_transform, download=True)

        train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, num_workers=0)

        return train_loader, test_loader

    # Model
    def loss_function(self, recon_x, x, mu, logvar):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    # Train
    def fit_train(self, epoch):
        self.train()
        LOGGER.info(f"Epoch: {epoch+1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(inputs)

            loss = self.loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            samples_cnt += inputs.size(0)

            if batch_idx%50 == 0:
                LOGGER.info(
                    "TRAIN epoch %s, batch %s/%s, loss %s",
                    epoch, batch_idx, len(self.train_loader), train_loss/samples_cnt)

        self.history["loss"].append(train_loss/samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                recon_batch, mu, logvar = self(inputs)
                val_loss += self.loss_function(recon_batch, inputs, mu, logvar).item()
                samples_cnt += inputs.size(0)

                if batch_idx == 0:
                    file_addr = os.path.join(
                        self.dump_folder,
                        f"epoch_{str(epoch)}_output.png"
                    )
                    save_image(recon_batch, file_addr, nrow=8)
                    file_addr = os.path.join(
                        self.dump_folder,
                        f"epoch_{str(epoch)}_input.png"
                    )
                    save_image(inputs, file_addr, nrow=8)

        LOGGER.info(
            "VAL epoch %s, loss %s",
            epoch, val_loss/samples_cnt)
        self.history["val_loss"].append(val_loss/samples_cnt)

        # sampling
        file_addr = os.path.join(
            self.dump_folder,
            f"epoch_{str(epoch)}_sampling.png"
        )
        save_image(self.sampling(), file_addr, nrow=8)

    # save results
    def save_history(self):
        file_addr = os.path.join(
            self.dump_folder,
            "history.csv"
        )
        pd.DataFrame(net.history).to_csv(file_addr)

if __name__ == "__main__":
    net = VAE("mnist")
    nb_epochs = 30
    net.init_model()
    for i in range(nb_epochs):
        net.fit_train(i + 1)
        net.test(i + 1)
    net.save_history()
