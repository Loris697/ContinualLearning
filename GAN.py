import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def plot_images(images, n_cols=4, title='Generated Images'):
    n_rows = len(images) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    idx = 0
    
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].imshow(images[idx].squeeze().numpy(), cmap='gray')
            axes[i, j].axis('off')
            idx += 1
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust layout to make space for the title
    plt.show()

class Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Discriminator will down-sample the input producing a binary output
        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)        
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)        
        self.fc4 = nn.Linear(in_features=32, out_features=out_features)
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        # Rehape passed image batch
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # Feed forward
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)                        
        x = self.fc3(x)
        x = self.leaky_relu3(x)        
        x = self.dropout(x)
        logit_out = self.fc4(x)
        
        return logit_out

class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Generator will up-sample the input producing input of size
        # suitable for feeding into discriminator
        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)        
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)        
        self.fc4 = nn.Linear(in_features=128, out_features=out_features)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        
        
    def forward(self, x):
        # Feed forward
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)        
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)        
        x = self.dropout(x)
        x = self.fc4(x)
        tanh_out = self.tanh(x)
        
        return tanh_out.view(-1, 1, 28, 28)

def real_loss(predicted_outputs, loss_fn, device):
    """
    Function for calculating loss when samples are drawn from real dataset
    
    Parameters
    ----------
    predicted_outputs: Tensor
                       predicted logits
            
    Returns
    -------
    real_loss: int
    """
    batch_size = predicted_outputs.shape[0]
    # Targets are set to 1 here because we expect prediction to be 
    # 1 (or near 1) since samples are drawn from real dataset
    targets = torch.ones(batch_size).to(device)
    real_loss = loss_fn(predicted_outputs.squeeze(), targets)
    
    return real_loss


def fake_loss(predicted_outputs, loss_fn, device):
    """
    Function for calculating loss when samples are generated fake samples
    
    Parameters
    ----------
    predicted_outputs: Tensor
                       predicted logits
            
    Returns
    -------
    fake_loss: int
    """
    batch_size = predicted_outputs.shape[0]
    # Targets are set to 0 here because we expect prediction to be 
    # 0 (or near 0) since samples are generated fake samples
    targets = torch.zeros(batch_size).to(device)
    fake_loss = loss_fn(predicted_outputs.squeeze(), targets)
    
    return fake_loss 

class GAN(nn.Module):
    def __init__(self, 
                 generator, 
                 discriminator,
                 d_optim = optim.Adam,
                 g_optim = optim.Adam,
                 d_lr = 0.0002,
                 g_lr = 0.002
                ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        # Instantiate optimizers
        self.d_optim = d_optim(self.discriminator.parameters(), lr=d_lr)
        self.g_optim = d_optim(self.generator.parameters(), lr=g_lr)
        # Instantiate the loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        #For Tensorboard logging
        #self.writer = SummaryWriter('runs/GAN_experiment')

    def forward(self, z = None):
        # Feed a latent vecor of size 100 to trained generator and get a fake generated image back
        if z is None:
            z = np.random.uniform(-1, 1, size=(1, 100))
            z = torch.from_numpy(z).float().cuda()
        
        return self.generator(z).view(1, 1, 28, 28).detach()

    def train_gan(self, dl, n_epochs = 100, device = 'cuda', verbose=False):
        print(f'Training on [{device}]...')
        
        # Generate a batch (say 16) of latent image vector (z) of fixed size 
        # (say 100 pix) to be as input to the Generator after each epoch of 
        # training to generate a fake image. We'll visualise these fake images
        # to get a sense how generator improves as training progresses
        z_size = 100
        fixed_z = np.random.uniform(-1, 1, size=(16, z_size))
        fixed_z = torch.from_numpy(fixed_z).float().to(device)          
        fixed_samples = []
        d_losses = []
        g_losses = []
        
        
        # Move discriminator and generator to available device
        self.discriminator = self.discriminator.to(device)
        self.generator = self.generator.to(device)
        
        for epoch in range(n_epochs):
            print(f'Epoch [{epoch+1}/{n_epochs}]:')
            # Switch the training mode on
            self.discriminator.train()
            self.generator.train()
            d_running_batch_loss = 0
            g_running_batch_loss = 0
            for curr_batch, (real_images, _) in enumerate(dl):
                # Move input batch to available device
                real_images = real_images.to(device)
                
                ## ----------------------------------------------------------------
                ## Train discriminator using real and then fake MNIST images,  
                ## then compute the total-loss and back-propogate the total-loss
                ## ----------------------------------------------------------------
                
                # Reset gradients
                self.d_optim.zero_grad()
                
                # Real MNIST images
                # Convert real_images value range of 0 to 1 to -1 to 1
                # this is required because latter discriminator would be required 
                # to consume generator's 'tanh' output which is of range -1 to 1
                real_images = (real_images * 2) - 1  
                d_real_logits_out = self.discriminator(real_images)
                d_real_loss = real_loss(d_real_logits_out, self.loss_fn, device)
                #d_real_loss = real_loss(d_real_logits_out, smooth=True)
                
                # Fake images
                with torch.no_grad():
                    # Generate a batch of random latent vectors 
                    z = np.random.uniform(-1, 1, size=(dl.batch_size, z_size))
                    z = torch.from_numpy(z).float().to(device)
                    # Generate batch of fake images
                    fake_images = self.generator(z) 
                # feed fake-images to discriminator and compute the 
                # fake_loss (i.e. target label = 0)
                d_fake_logits_out = self.discriminator(fake_images)
                d_fake_loss = fake_loss(d_fake_logits_out, self.loss_fn, device)
                #d_fake_loss = fake_loss(d_fake_logits_out)
                # Compute total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                # Backpropogate through discriminator
                d_loss.backward()
                self.d_optim.step()
                # Save discriminator batch loss
                d_running_batch_loss += d_loss
                
                ## ----------------------------------------------------------------
                ## Train generator, compute the generator loss which is a measure
                ## of how successful the generator is in tricking the discriminator 
                ## and finally back-propogate generator loss
                ## ----------------------------------------------------------------
    
                # Reset gradients
                self.g_optim.zero_grad()
                
                # Generate a batch of random latent vectors
                #z = torch.rand(size=(dl.batch_size, z_size)).to(device)
                z = np.random.uniform(-1, 1, size=(dl.batch_size, z_size))
                z = torch.from_numpy(z).float().to(device)       
                # Generate a batch of fake images, feed them to discriminator
                # and compute the generator loss as real_loss 
                # (i.e. target label = 1)
                fake_images = self.generator(z) 
                g_logits_out = self.discriminator(fake_images)
                g_loss = real_loss(g_logits_out, self.loss_fn, device)
                #g_loss = real_loss(g_logits_out)
                # Backpropogate thorugh generator
                g_loss.backward()
                self.g_optim.step()
                # Save discriminator batch loss
                g_running_batch_loss += g_loss
                
                # Display training stats for every 200 batches 
                if curr_batch % 400 == 0 and verbose:
                    print(f'\tBatch [{curr_batch:>4}/{len(dl):>4}] - d_batch_loss: {d_loss.item():.6f}\tg_batch_loss: {g_loss.item():.6f}')
                
            # Compute epoch losses as total_batch_loss/number_of_batches
            d_epoch_loss = d_running_batch_loss.item()/len(dl)
            g_epoch_loss = g_running_batch_loss.item()/len(dl)
            d_losses.append(d_epoch_loss)
            g_losses.append(g_epoch_loss)
            
            # Display training stats for every 200 batches 
            print(f'epoch_d_loss: {d_epoch_loss:.6f} \tepoch_g_loss: {g_epoch_loss:.6f}')
            
            # Generate fake images from fixed latent vector using the trained 
            # generator so far and save images for latter viewing
            self.generator.eval()

            if epoch % 10 == 9:
                generated_images = self.generator(fixed_z).detach().cpu()
                plot_images(generated_images, title='GAN Generated Images at epoch ' + str(epoch + 1)) # Create a grid of images
            #self.writer.add_image('Generated Images', grid, global_step=step)
         
        return d_losses, g_losses