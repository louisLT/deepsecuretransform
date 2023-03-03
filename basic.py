import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
# import torchvision
import logging
import numpy as np
import cv2

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s',
                    level=logging.INFO,
                    force=True)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
LOGS_DIR = os.environ["LOGS_DIR"]
DUMPS_DIR = os.path.join(LOGS_DIR, "dumps")

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        existing_folders = sorted(os.listdir(DUMPS_DIR))
        if existing_folders:
            last_version = int(existing_folders[-1][-2:])
        else:
            last_version = -1
        self.dump_folder = os.path.join(DUMPS_DIR, "version_%s" % str(last_version + 1).zfill(3))
        os.makedirs(self.dump_folder)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # LOGGER.info("test log !")
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = x.view(x.size(0), -1)
        y = self.decoder(self.encoder(y))
        y = y.view(x.shape)
        image_arrays = {"x": x.cpu().detach().numpy(),
                        "y": y.cpu().detach().numpy()}
        LOGGER.info("batch idx %s %s", self.dump_folder, str(batch_idx).zfill(3))
        dump_dir = os.path.join(self.dump_folder, str(batch_idx).zfill(3))
        os.makedirs(dump_dir)
        LOGGER.info("dumping to directory %s", dump_dir)
        for str_ in ["x", "y"]:
            for i_sample in range(x.shape[0]):
                img_ = np.rollaxis(image_arrays[str_][i_sample], 0, 3)
                img_ = (np.clip(img_, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        dump_dir,
                        "img_%s_%s.png" % (str(i_sample).zfill(3), str_)),
                        img_)


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

# setup data
dataset = MNIST(PATH_DATASETS, download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, batch_size=16)
val_loader = utils.data.DataLoader(dataset, batch_size=4)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=20,
                     limit_val_batches=1,
                     max_epochs=2,
                     default_root_dir=LOGS_DIR,
                     logger=False)
trainer.fit(model=autoencoder,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)