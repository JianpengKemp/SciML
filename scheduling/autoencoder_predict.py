from logging import logProcesses
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from torchsummary import summary
from sklearn.model_selection import train_test_split


raw_data = pd.read_csv('/Users/jianpeng/Desktop/Sciml/data/diabetes.csv')
device = ('cuda' if torch.cuda.is_available() else 'cpu')

x_train, x_test, y_train, y_test = train_test_split(raw_data.drop('Outcome',axis=1), 
                                                    raw_data.Outcome, test_size=0.15, stratify=raw_data.Outcome)


x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)
positives = x_train[y_train == 1]
negatives = x_train[y_train == 0]

negatives= torch.from_numpy(negatives).to(device)

class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 100),
            nn.Tanh(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 75),
            nn.Tanh(),
            nn.BatchNorm1d(75),
            nn.Linear(75, 50),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(50, 25),
            nn.ReLU(True),
            nn.BatchNorm1d(25),
            nn.Linear(25, enc_shape),
            nn.ReLU(True),
        )
        
        self.decode = nn.Sequential(
            nn.Linear(enc_shape,25),
            nn.ReLU(True),
            nn.BatchNorm1d(25),
            nn.Linear(25,50),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(50,75),
            nn.ReLU(True),
            nn.BatchNorm1d(75),
            nn.Linear(75,100),
            nn.Tanh(),
            nn.BatchNorm1d(100),
            nn.Linear(100,in_shape),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def train(model, error, optimizer, n_epochs, x):
    model.train()
    losses = []
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
        
        if epoch % int(0.1*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')


model = Autoencoder(in_shape=negatives.shape[1], enc_shape=7).double().to(device)
print(model)
error = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

train(model, error, optimizer, 1000, negatives)
torch.save(model.encode, "encoder.pt")
torch.save(model.decode, "decoder.pt")

encoder = torch.load("encoder.pt")
decoder = torch.load("decoder.pt")

with torch.no_grad():
    encoded = encoder(negatives)
    decoded = decoder(encoded)
    mse = error(decoded, negatives).item()
    enc = encoded.cpu().detach().numpy()
    dec = decoded.cpu().detach().numpy()

print(f'Root mean squared error: {np.sqrt(mse):.4g}')


x_train = torch.from_numpy(x_train).to(device)
x_train_transform = encoder(x_train)
y_train = torch.unsqueeze(torch.from_numpy(y_train.to_numpy()),dim=1).double().to(device)
train_data = torch.hstack((x_train_transform, y_train))
train_loader = torch.utils.data.DataLoader(train_data, batch_size= 10, shuffle=True)


class Predictor(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape):
        super(Predictor, self).__init__()
        
        self.predict = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Linear(64,1)
        )
        
    def forward(self, x):
        y = self.predict(x)
        return y

def train(model, error, optimizer, n_epochs, dataloader):
    model.train()
    losses = []
    for epoch in range(1, n_epochs + 1):
        for batch in dataloader:
            features, label = batch[:, :-1], batch[:, -1]
            optimizer.zero_grad()
            output = model(features)
            output = torch.squeeze(output,1)
            loss = error(output, label)
            loss.backward(retain_graph=True) ## why this retain_graph has to be set True?
            optimizer.step()
            losses.append(loss.item())
        
        if epoch % int(0.1*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')

model = Predictor(7).double().to(device)
print(model)
error = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

train(model, error, optimizer, 1000, train_loader)
torch.save(model.encode, "encoder.pt")
torch.save(model.decode, "decoder.pt")
