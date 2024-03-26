import io
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

# TODO:
# (1) - Squeeze data in the loading process
# (2) - Implement Distributed Data Parallel to speedup training. 

# Random masking transform of input data.
class RandomMasking(object):
    def __init__(self, masking_ratio=0.25):
        self.masking_ratio = masking_ratio

    def __call__(self, sample):
        n = sample.shape[1]        
        indexes = torch.randperm(n)[:int(self.masking_ratio * n)] # Generates random permutation of integers from 0 to n-1, select first 256 in order to get 256 random unique integers        
        sample[0][indexes] = 0
        return sample

class ExtractedFeaturesDataset(Dataset):
    def __init__(self, data_dir='../ViT/feature_extraction/features/PC2022/PC2022_features_dictionary.pt', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform    
        with open(self.data_dir, 'rb') as f: 
            buffer = io.BytesIO(f.read()) # extremely faster
            feature_dictionary = torch.load(buffer, map_location=torch.device('cuda:0'))
        self.image_features = [e[1] for key in list(feature_dictionary.keys()) for e in feature_dictionary[key]]
    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        features = self.image_features[idx]
        label = features.clone()#.detach()
        if self.transform:
            features = self.transform(features) # Apply random masking
        return features, label

class VanillaAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 768),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(768, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),            
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU()            
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 768),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(768, 1024),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

# DataLoader is used to load the dataset 
# for training
batch_size = 64 * 1024

dataset = ExtractedFeaturesDataset(transform=RandomMasking(masking_ratio=0.25))
loader = torch.utils.data.DataLoader(dataset = dataset,batch_size = batch_size, shuffle = False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VanillaAE()
model = model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-1, weight_decay = 1e-8)

epochs = 50
outputs = []
losses = []
total_steps = int(dataset.__len__() / batch_size)

for epoch in range(epochs):
    iteration = 1
    
    for (image, label) in loader:
      image = image.squeeze(1)
      label = label.squeeze(1) # TODO FIX in the loading process.
      
      if iteration % 10 == 0:  
        print('   Step no. ', '['+str(iteration)+'/'+str(total_steps)+']')

      # Output of Autoencoder
      reconstructed = model(image)
       
      # Calculating the loss function
      loss = loss_function(reconstructed, label)
       
      # The gradients are set to zero,
      # the gradient is computed and stored.
      # .step() performs parameter update
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Storing the losses in a list for plotting
      losses.append(loss.cpu())
      iteration += 1
    print('Epoch ['+ str(epoch + 1) + '/'+ str(epochs) + '] - Loss: ', loss.item()) # format
    #outputs.append((epochs, image, reconstructed))
 
torch.save(model.state_dict(), '64/d_AE_64bottleneck_checkpoint-epoch-{}.pth'.format(epochs))

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

epoch_loss = []
sum = []
for iteration in range(len(losses)):
    if iteration % total_steps == 0:
        epoch_loss.append(np.average(sum))
        sum = []
    sum.append(losses[iteration].detach().numpy())

plt.plot(epoch_loss)
plt.savefig('64/denoising_AE_loss_x_epoch.pdf')