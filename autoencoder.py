import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import os
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


'''
# @Author: Chen Wei
# - Code modified from tutorial & my own other research project: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
# - Idea: This is an VAE trained on MNIST, which can also performs image compression.
# - Disclaimer: the reason I use MNIST as dataset is that we don't need to collect extra data, the disadvantage of this
# is the dimension of the layers of CNN need to be modified if we wants to perform the vae we trained to reconstruct a 
# RGB image. i.e. need to change input channel to 3 
# - Although not being part of the fractal image compression, the reason why I include VAE here is that we could do some 
# comparison based on different model
'''
data_dir = "data"

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(256),
     transforms.ToTensor(),
])

test_transform = transforms.Compose([
  #   transforms.Resize(256),
  #   transforms.CenterCrop(256),
     transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

print("Classes: ")
class_names = train_dataset.classes
print(class_names)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        # x - (4, 3, 32,32)   4 - # sample   3 - rgb  32* 32- image
        # (W - F + 2P)/S + 1
        # W - input    F- filter  P - padding  S- stride
        super(VariationalEncoder, self).__init__()
        # 1 - input channel/grey  change to 3 for rgb
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3 * 3 * 32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)


    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_loss(self, x, x_hat):
        return ((x - x_hat) ** 2).sum()

    def kl_loss(self):
        return vae.encoder.kl

    def combined_loss(self, x, x_hat):
        return self.reconstruction_loss(x, x_hat) + self.kl_loss()




### Training function
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        '''x_hat, z = vae(x)'''
        x_hat = vae(x)
        # Evaluate loss
        loss = vae.combined_loss(x, x_hat)

        # Backward pass
        optimizer.zero_grad()   # clear gradient
        loss.backward()    # calculate gradient
        optimizer.step()   # update
        # Print batch loss
     #   print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

### Testing function
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)    #latent dimension
            #Todo: plot hidden deminsion as ...

            # Decode data
            x_hat = vae(x)
            loss = vae.combined_loss(x, x_hat)
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
         latent_img = encoder(img)  # here is the compressed image
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')

      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def eval_image(vae):
    vae.eval()

    with torch.no_grad():
        # sample latent vectors from the normal distribution
        latent = torch.randn(128, 4, device=device)

        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(20, 8.5))
        show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
        plt.show()

def plot_latent(encoder,decoder,n=10):
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
        #    rec_img = decoder(encoder(img))
            latent_img = encoder(img)  # here is the compressed image
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(latent_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Latent images')
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)

    vae = VariationalAutoencoder(latent_dims=4)

    lr = 1e-3

    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    vae.to(device)

    num_epochs = 50

    # Train
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(vae, device, train_loader, optim)
    #     val_loss = test_epoch(vae, device, valid_loader)
    #
    #     print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss,
    #                                                                           val_loss))
    #     plot_ae_outputs(vae.encoder, vae.decoder, n=10)
    # torch.save(vae.state_dict(), "model/vae_mnist.pth")


    # Evalutate
    vae.load_state_dict(torch.load("model/vae_mnist.pth"))
   # plot_latent(vae.encoder, vae.decoder, n=10)
   # Todo: modify it so it can plot latent, hard, as cannot separate encoder & decoder after loading weight
