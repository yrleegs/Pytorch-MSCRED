import torch
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
from model.mscred import MSCRED
from utils.data import load_data
import matplotlib.pyplot as plt
import numpy as np
import os

def train(dataLoader, model, optimizer, epochs, device):
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum,n = 0.0, 0
        for x in tqdm(dataLoader):
            x = x.to(device)
            target = x.transpose(0, 1)[-1]
            l = F.mse_loss(model(x), target, reduction='mean')
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, train_l_sum/n))

def test(dataLoader, model):
    print("------Testing-------")
    index = 800
    reconstructed_data_path = "./data/matrix_data/reconstructed_data/"
    os.makedirs(reconstructed_data_path, exist_ok=True)

    with torch.no_grad():
        for x in dataLoader:
            x = x.to(device)
            reconstructed_mats = model(x).cpu().numpy()
            for i in range(reconstructed_mats.shape[0]):
                reconstructed_mat = np.expand_dims(reconstructed_mats[i], 0)
                path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
                np.save(path_temp, reconstructed_mat)
                index += 1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_file_path = './model.pth'
    batch_size = 16
    epochs = 10
    channels = 3
    hidden_size = 256
    learning_rate = 0.0002
    
    dataLoader = load_data(batch_size)

    mscred = MSCRED(channels, hidden_size).to(device)

    optimizer = torch.optim.Adam(mscred.parameters(), lr=learning_rate)
    train(dataLoader["train"], mscred, optimizer, epochs, device)
    torch.save(mscred.state_dict(), save_file_path)

    mscred.load_state_dict(torch.load(save_file_path))
    test(dataLoader["test"], mscred)