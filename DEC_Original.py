'''
Script trains a DEC model according to 	arXiv:1511.06335 
In script, 
    data is loaded and preprocessed
    model parameters are defined
    autoecnoder is pretrained 
    KMeans is fitted to representation learned by autoencoder
    DEC model is fitted to align predictions of encoder from autoencoder with cluster centers
'''

'''
Necessary libraries are imported
'''
import numpy as np
import torch.optim as optim
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import datetime
from PIL import Image
import csv
'''
Self written modeules for DEC
'''
from Modules import Cluster
from Modules import KMeans
from Modules import AE_VGG
from Modules.Utils import cluster_accuracy, target_distribution
print('Finished import of libraries')


'''
Transformations for images are defined
More augmentations for training images than for test and validation images
Transformations for test- and validation images are identical
'''
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])
transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])

'''
Get image paths 
Split image paths in training- test- and validation paths
Image paths are basis for class ImageDatset which loads images in batches
'''
with open('ListOfImages.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)  
im_paths  = np.array(data).reshape(len(data[0]))     
np.random.shuffle(im_paths)
train_paths, rest = np.split(im_paths, [int(0.8*len(im_paths))])
test_paths, val_paths = np.split(rest, [int(0.5*len(rest))])

'''
Class which loads images
Is instance of torch.utils.data.Dataset
Call class with: 
    File_Name: List or np.array of paths to images
    transforms: instance of torchvision.transforms.Compose
'''
class ImageDataset(Dataset):
    def __init__(self,
                File_Name, 
                transform=False):
        self.image_paths = File_Name
        self.transform = transform
    def __len__(self):
        return(len(self.image_paths))
    def __getitem__(self, idx):
        im_path = '/workspace/'+self.image_paths[idx].replace('\'', '')
        im = Image.open(im_path)
        im = np.array(im)
        if(self.transform is not None):
            im = self.transform(im)
        return im

'''
Create ImageDatasets for train-, validation and test images
Put ImageDatsets into correpsonding torch.utils.data.DataLoader
'''
train_data = ImageDataset(train_paths, transform_train)
test_data = ImageDataset(test_paths, transform_test)
val_data = ImageDataset(val_paths, transform_val)
train_loader = DataLoader(train_data, batch_size =8, shuffle = True, num_workers = 0, drop_last = True)
test_loader = DataLoader(test_data, batch_size = 8, shuffle = False, num_workers = 0, drop_last = True)
val_loader = DataLoader(val_data, batch_size=8, shuffle = True, num_workers=0, drop_last = True)


'''
Function which predicts the cluster an image belongs to 
Function is called in training loop 
Funtion returns tensor which contains most liekly cluster for evey image from tarining data
Pass: 
    Model: DEC model, instance of torch.nn.module
    Device: String, specifies on whioch device to work
Returns: 
    Most likely cluster for any datapoint
'''
def predict(model: nn.Module, 
            device: str = 'cpu'):
    features = []
    for ims in iter(train_loader):
        ims = ims.to(device)
        out = model.forward(ims)
        features.append(out.detach().cpu())
    return torch.cat(features).max(1)[1]

'''
Function for validaion of pretraining results
Returns loss on validation data
Can be called after every epoch for seeing overfitting early
Pass:
    model: Trained model
    val_loader: Data loader which shall be used for validation
    criterion: Loss function 
'''
def pretrain_validation(model: nn.Module,
                       val_loader: DataLoader,
                      criterion: nn.Module):
    model.eval()
    val_loss = 0
    for ims in iter(val_loader):
        ims = ims.to('cuda')
        bn, rec = model.forward(ims)
        loss = criterion(rec, ims)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

'''
Function which pretrains autoencoder
Function returns name of best model trained
After every epoch:
    Model is saved
    Results are written to text file
    bottle neck and reconstruted images for one batch of test set are saved to csv-file
After training has finished, losses are written to csv-file
Pass: 
    Autoencoder architetcure to be trained
    Numer of training epochs
    Thereshold for early stopping
'''
def pretrain(autoencoder: nn.Module,
             num_epochs: int = 2, 
             device: str = 'cpu', 
             threshold_early_stopping: int = 5):
      optimizer = SGD(autoencoder.parameters(), lr=0.01, momentum=0.9)
      criterion = nn.MSELoss()
      losses = []
      train_losses = []
      best_loss = 10000000
      val_losses = []
      best_epoch = 0
      for epoch in range(num_epochs):
        print('\n\nepoch: '+ str(epoch+1))
        autoencoder.train()
        running_loss = 0
        for ims in iter(train_loader):
            ims = ims.to(device)
            optimizer.zero_grad()
            bn, reconstructed = autoencoder.forward(ims)
            loss = criterion(reconstructed, ims)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            running_loss+=loss.item()
        autoencoder.eval()
        train_loss = running_loss/len(train_loader)
        losses.append(train_loss)
        val_loss = pretrain_validation(autoencoder, val_loader, criterion)
        val_losses.append(val_loss)
        print('Epoch: {}/{}.. '.format(epoch+1, num_epochs),
            'training_loss: {:.3f}..'.format(train_loss))
        f = open('PathToResults/Results.txt', 'a')
        now = datetime.datetime.now().strftime('%H_%M_%S')
        f.write(f'\n\n\n\nPretraining: \nEpoch: {epoch}/{num_epochs} \ntime: {now}. \ntraining_loss: {running_loss}. \nvalidation_loss: {val_loss}.')    
        file_name = f'PathToModels/Autoencoder_DEC_pretraining_epoch_{epoch}_loss_{int(val_loss*1000)}_at_{now}.pth'
        torch.save(autoencoder.state_dict(), file_name)
        bn, rec =  autoencoder.forward(next(iter(test_loader)).to(device))
        name = f'PathToResults/Rec_pretraining_epoch_{epoch}_at_{now}.csv'
        rec.cpu().detach().numpy().tofile(name,  sep = ',')
        name = f'PathToResults/BN_pretraining_epoch_{epoch}_at_{now}.csv'
        bn.cpu().detach().numpy().tofile(name, sep = ',')
        if(loss<best_loss):
            best_loss = loss
            best_epoch = epoch
            best_model = file_name
        elif(epoch-best_epoch > threshold_early_stopping):
            print('Training stopped')
            break  
        running_loss = 0
        autoencoder.train()
      np.array(val_losses).tofile(f'PathToResults/Validation_lossses_pretraining.csv', sep = ',')
      np.array(losses).tofile(f'PathToResults/Training_lossses_pretraining.csv', sep = ',')
      print(f'Pretraining finished: Best model: {best_model}')
      return(best_model)
  

'''
Function for validaion of pretraining results
Returns loss on validation data
Can be called after every epoch for seeing overfitting early
Pass:
    model: Trained model
    val_loader: Data loader which shall be used for validation
    criterion: Loss function 
'''
def validation_train(model: nn.Module,
                    val_loader: DataLoader,
                    criterion: nn.Module):
        val_loss = 0.0
        model.eval()
        for ims in iter(val_loader):
            ims = ims.to(device)
            output = model.forward(ims)#Returns fitness of every value to corresponsing cluster
            target = target_distribution(output).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            val_loss += loss.item()
        val_loss/=len(val_loader)
        return(val_loss)

'''
Function which trains original DEC model 
Returns: 
    trained model
    K-Means model 
    Cluster centers of K-Means model
Within function: 
    K-Means model is fitted to reduced representation  of initiall autoencoder
    Cluster centers are extracted from K-Means model and set as parameters in DEC model
    DEC model is fitted to align predictions with cluster centers
Pass: 
    model: DEC model to be fitted
    optimizer: Optimizer to be used in training
    epochs: Number of epochs for training
    device: Device, data shall be processed on
    n_clusters: Number of clusters for K-Means model
    stopping_delta: Minimum difference in previous and current prediction before training is stopped 
'''
def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    device: str = 'cpu',
    n_clusters: int = 3,
    stopping_delta: float = 0.05,
    ) -> None:

    best_model = ''
    best_loss = np.inf
    #1) Initialize KMeans
    kmeans = KMeans.Kmeans(n_clusters=model.cluster_number)
    model.train()
    features = []
    actual = []
    #1a) predict reduced presentation by models encoder for every batch in train loader
    for ims in iter(test_loader):
        if (isinstance(ims, tuple) or isinstance(ims, list)) and len(ims) == 2:
            ims, value = ims  # if we have a prediction label, separate it to actual
            actual.append(value)
        ims = ims.to(device)
        features.append(model.encoder(ims)[0].detach().cpu())
    #1b) Fit kmeans to predictions
    kmeans.fit(torch.cat(features).numpy())
    np.array(kmeans.centroids).tofile('PathToResults/Centroids.csv', sep= ',')
    #1c) Get predicted cluster from kmeans
    clusters = kmeans.get_cluster(torch.cat(features).numpy())
    last_clusters = torch.tensor(np.copy(clusters), dtype=torch.long)
    #1d) cluster centers from kmenas and assign them to model's assignment-cluster_centers
    cluster_centers = torch.tensor(
        kmeans.centroids, dtype=torch.float, requires_grad=True
    ).to(device)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)


    #2) Train model
    loss_function = nn.KLDivLoss(size_average=False)
    val_losses = []
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for ims in iter(train_loader):
            ims = ims.to(device)
            output = model.forward(ims)#Returns fitness of every value to corresponsing cluster
            target = target_distribution(output).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
        running_loss /= len(train_loader)
        f = open('PathToResults/results.txt', 'a')
        now = datetime.datetime.now().strftime('%H_%M_%S')
        f.write(f'\n\n\n\nTraining: \nEpoch: {epoch}/{epochs} \ntime: {now}. \ntraining_loss: {running_loss}.')    
        name = f'PathToModels/Model_DEC_epoch_{epoch}_loss_{int(running_loss*1000)}_at_{now}.pth'
        torch.save(model.state_dict(), name)
        clusters = predict(model,
                           device = device)
        losses.append(running_loss)
        val_loss = validation_train(model ,
                                    val_loader, 
                                    loss_function)
        val_losses.append(val_loss)
        
        delta_label = (float((clusters != last_clusters).float().sum().item())/ last_clusters.shape[0])
        
        if stopping_delta is not None and delta_label < stopping_delta:
            print('Early stopping')
            break
        
        last_clusters = clusters
        if (running_loss < best_loss): 
             best_model = name
             best_loss = running_loss
        ims = next(iter(test_loader))
        out = model.forward(ims.to(device))
        ims.cpu().detach().numpy().tofile(f'PathToResults/ims_epoch_{epoch}.csv', sep = ',')
        out.cpu().detach().numpy().tofile(f'PathToResults/out_epoch_{epoch}.csv', sep = ',')
        bn = model.encoder.forward(ims.to(device))[0]
        bn.cpu().detach().numpy().tofile(f'PathToResults/bn_epoch_{epoch}.csv', sep = ',')
        
        

        
    model.load_state_dict(torch.load(best_model))
    return(model, kmeans, kmeans.centroids)  
        



'''
Work here
Define parameters for model
Call functions accordingly
1: Pretraining of autoencoder
    --> If there already is an pretrained autoencoder coment pretrain and define name of model as best_model
    --> In order to ensure, autoencoder is trained and loaded correctly, one batch from test data is used for testing, predictions are stored to csv

2: Train DEC model
    --> Create model using pretrained autoencoder
    -->Define parameters for training
    --> Train
'''

device = 'cuda:0'
'''
Training of autoencoder
Here, convolutional autoencoder following VGG16 architecture is used
You can vary the number of features in the bottleneck of the autoencoder
'''
autoencoder = AE_VGG.Autoencoder(num_features = 10).to(device)
print("Pretraining stage.")
best_model = pretrain(autoencoder, 
             num_epochs = 5, 
             device = device, 
             threshold_early_stopping = 5)

autoencoder.load_state_dict(torch.load(best_model, map_location = device))

#Test if autoencoder performs sufficiently
out = autoencoder.forward(next(iter(test_loader)).to(device))
print(out)
bn = out[0]
im = out[1]
bn.cpu().detach().numpy().tofile('PathToResults/bn_autoencoder_10_features_pretraining.csv', sep = ',')
im.cpu().detach().numpy().tofile('PathToResults/im_autoencoder_10_features_pretraining.csv', sep = ',')

#Here begins DEC train
print("DEC stage.")
model = Cluster.DEC(cluster_number=4, 
                        hidden_dimension = autoencoder.num_features, 
                        encoder=autoencoder).to(device)
dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 200
model, kmeans, kmeans.centroids = train(
    model = model,
    optimizer = dec_optimizer,
    epochs = num_epochs,
    device = device,
    n_clusters = 4,
    )
    
