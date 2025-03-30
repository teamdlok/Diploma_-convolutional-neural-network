
from skimage.io import imread
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.data import DataLoader
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time
import cv2
from matplotlib import rcParams
from PIL import Image
from torchvision.utils import save_image

train_acc_history = []
val_acc_history = []
epoch_history = []

images = []
lesions = []

root = 'PH2Dataset'

for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
    if root.endswith('_Dermoscopic_Image'):
        images.append(imread(os.path.join(root, files[0])))
    if root.endswith('_lesion'):
        lesions.append(imread(os.path.join(root, files[0])))


size = (384, 384)
X = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]


X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
print(f'Loaded {len(X)} images')

len(lesions)


plt.figure(figsize=(18, 6))
for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.axis("off")
    plt.imshow(X[i])

    plt.subplot(2, 6, i+7)
    plt.axis("off")
    plt.imshow(Y[i])
plt.show();


ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [55, 65])

print(len(tr), len(val), len(ts))


batch_size = 10
data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])), 
                     batch_size=batch_size, shuffle=True)
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
                      batch_size=batch_size, shuffle=True)
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
                     batch_size=batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

rcParams['figure.figsize'] = (15,4)

class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU()
        )
        self.pool0 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True) # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True) # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True) # 32 -> 16

        # bottleneck
        self.bottleneck_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU()
        )
        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bottleneck_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.bottleneck_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU()
        )
        
        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU()
        )
        self.upsample1 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU()
        )
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU()
        )
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum = 0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e0, id1 = self.pool0(e0)
        e1 = self.enc_conv1(e0)
        e1, id2 = self.pool1(e1)
        e2 = self.enc_conv2(e1)
        e2, id3 = self.pool2(e2)
        e3 = self.enc_conv3(e2)
        e3, id4 = self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv0(e3)
        b1, id5 = self.bottleneck_pool(b)
        b2 = self.bottleneck_unpool(b1, id5)
        b3 = self.bottleneck_conv1(b2)

        # decoder
        d0 = self.upsample0(b3, id4)
        d0 = self.dec_conv0(d0)
        d1 = self.upsample1(d0, id3)
        d1 = self.dec_conv1(d1)
        d2 = self.upsample2(d1, id2)
        d2 = self.dec_conv2(d2)
        d3 = self.upsample3(d2, id1)
        d3 = self.dec_conv3(d3) # no activation
        return d3
    
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte() # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou 

def bce_loss(y_real, y_pred):
  return torch.mean(y_pred.clamp(min=0) - y_real*y_pred + torch.log(1 + torch.exp(-torch.abs(y_pred))))

t1 = torch.randn(64, 1, 256, 256)
t2 = torch.randn(64, 1, 256, 256)
print(nn.BCEWithLogitsLoss()(t1, t2), bce_loss(t1, t2))

def train(model, opt, loss_fn, epochs, data_tr, data_val):
    X_val, Y_val = next(iter(data_val))
    history = []

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        train_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            opt.zero_grad()
            # set parameter gradients to zero

            # forward
            Y_pred = model.forward(X_batch) # forward-pass
            loss = loss_fn(Y_batch, Y_pred)
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate loss to show the user
            train_loss += loss / len(data_tr)
            train_acc = iou_pytorch(torch.sigmoid(Y_pred)>0.5, Y_batch).mean().item()
            X_batch = X_batch.cpu()
            Y_batch = Y_batch.cpu()
            loss = loss.cpu()
            Y_pred = Y_pred.cpu()
            del X_batch, Y_batch, Y_pred
        toc = time()
        print('train loss: %f' % train_loss)

        # show intermediate results
        val_loss = 0
        model.eval()  # testing mode
        with torch.set_grad_enabled(False):
          Y_val = Y_val.to(device)
          Y_hat = model(X_val.to(device)) # detach and put into cpu
          val_loss = loss_fn(Y_val, Y_hat)
          val_acc = iou_pytorch(torch.sigmoid(Y_hat)>0.5, Y_val).mean().item()
          X_val = X_val.cpu()
          Y_val = Y_val.cpu()
          Y_hat = Y_hat.detach().cpu()

        history.append((train_loss, train_acc, val_loss, val_acc))
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        epoch_history.append(epoch)

        if epoch % 1 == 0:

            plt.figure(figsize=(15, 9))
            plt.plot( epoch_history,train_acc_history, label="train_acc")
            plt.plot( epoch_history,val_acc_history, label="val_acc")
            plt.legend(loc='best')
            plt.xlabel("epochs")
            plt.ylabel("acc")
            plt.show()

            # Visualize tools
            clear_output(wait=True)
            for k in range(6):
                plt.subplot(3, 6, k+1)
                plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')

                x_val_numpy_array = np.rollaxis(X_val[k].numpy(), 0, 3)
                # img = Image.fromarray((x_val_numpy_array * 255).astype(np.uint8))
                # img.save("test_img_from_ai.png")
                x_valid_white_px = np.sum((x_val_numpy_array * 255).astype(np.uint8) == 255)
                # img_two = X_val[k]
                # save_image(img_two, f'test_img_from_ai{k}.png')
                # x_valid_white_px = np.sum(X_val[k].numpy() == 255)
                print(f"10 meters picture white px - {x_valid_white_px}")

                plt.title('Real 10 meters')
                plt.axis('off')

                plt.subplot(3, 6, k+7)
                plt.imshow(Y_hat[k, 0]>0.5, cmap='gray')

                y_hat_numpy_array = Y_hat[k, 0]>0.5
                y_hat_white_px = np.sum((y_hat_numpy_array.numpy() * 255).astype(np.uint8) == 255)
                print(f"AI picture white px - {y_hat_white_px}")
                print(np.unique(y_hat_white_px))

                plt.title('Output AI')
                plt.axis('off')

                plt.subplot(3, 6, k+13)
                plt.imshow(np.rollaxis(Y_val[k].numpy(), 0, 3), cmap='gray')

                y_val_numpy_array = np.rollaxis(Y_val[k].numpy(), 0, 3)
                y_valid_white_px = np.sum((y_val_numpy_array * 255).astype(np.uint8) == 255)
                print(f"30 meters picture white px - {y_valid_white_px} \n")

                plt.title('Output 30 meters')
                plt.axis('off')

            plt.suptitle('%d / %d - train_loss: %f, val_loss: %f' % (epoch+1, epochs, train_loss, val_loss))
            plt.show()

    return history

def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [ X_batch for X_batch, _ in data]
    return np.array(Y_pred)

def score_model(model, metric, data):
    model.eval()  # testing mode
    scores = 0
    for X_batch, Y_label in data:
        X_batch = X_batch.to(device)
        Y_label = Y_label.to(device)
        Y_pred = (torch.sigmoid(model(X_batch))) > 0.5        
        scores += metric(Y_pred, Y_label.to(device)).mean().item()
        Y_pred.cpu()
        X_batch.cpu()
        del X_batch, Y_pred
    return scores/len(data)

segnet = SegNet().to(device)

max_epochs = 20
optim = torch.optim.Adam(segnet.parameters())
train(segnet, optim, bce_loss, max_epochs, data_tr, data_val)

score_model(segnet, iou_pytorch, data_val)