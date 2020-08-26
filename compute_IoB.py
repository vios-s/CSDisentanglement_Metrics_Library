import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from metrics.IoB_models import Cont_Trainer, Sty_Trainer
import os
import argparse

#Load the data directory and saving path
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='', help='Path to the data file.')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
parser.add_argument('--save', type=str, default='', help='Path to save the results.')
opts = parser.parse_args()

gpu_num = opts.gpu
dir_root = opts.root + '/'

device = torch.device("cuda:"+gpu_num if torch.cuda.is_available() else "cpu")
dir_root = opts.root + '/'
train_cont_root = 'content_train.npz'
train_sty_root = 'style_train.npz'
train_img_root = 'images_train.npz'
test_cont_root = 'content_test.npz'
test_sty_root = 'style_test.npz'
test_img_root = 'images_test.npz'
result_directory = opts.save+'IoB_result.txt'

#Load the data to train IoB models
train_content = np.load(dir_root+train_cont_root)['arr_0']
train_style = np.load(dir_root+train_sty_root)['arr_0']
train_images = np.load(dir_root+train_img_root)['arr_0']
test_content = np.load(dir_root+test_cont_root)['arr_0']
test_style = np.load(dir_root+test_sty_root)['arr_0']
test_images = np.load(dir_root+test_img_root)['arr_0']

train_num_samples = train_images.shape[0]
test_num_samples = test_images.shape[0]

#Transpose data for pytorch
train_content = torch.from_numpy(train_content)
train_style = torch.from_numpy(train_style)
train_images = torch.from_numpy(train_images)
train_Bias_content = torch.ones_like(train_content)
train_Bias_style = torch.ones_like(train_style)

test_content = torch.from_numpy(test_content)
test_style = torch.from_numpy(test_style)
test_images = torch.from_numpy(test_images)
test_Bias_content = torch.ones_like(test_content)
test_Bias_style = torch.ones_like(test_style)

#Train IoB models
#define the Content AutoEncoders
epoch = 40
batch_size = 10
num_itr = int(train_num_samples / batch_size)

Content_AE = Cont_Trainer()

#Train the Content AutoEncoder
print('Start training Content Autoencoder...')
Content_AE.to(device)
for ep in range(epoch):
    index = torch.randperm(train_num_samples)
    for i in range(num_itr):
        content = train_content[index[i*batch_size:(i+1)*batch_size],:,:,:].float().to(device)
        target = train_images[index[i*batch_size:(i+1)*batch_size],:,:,:].float().to(device)
        MSE_loss = Content_AE.AE_update(content, target)
    print('Epoch: %d, MSE_loss: %f'% (ep, MSE_loss))
print('Content Autoencoder is trained!')


Bias_AE = Cont_Trainer()
#Train the Content Biase AutoEncoder
print('Start training Content Bias Autoencoder...')
Bias_AE.to(device)
for ep in range(epoch):
    index = torch.randperm(train_num_samples)
    for i in range(num_itr):
        content = train_Bias_content[index[i*batch_size:(i+1)*batch_size],:,:,:].float().to(device)
        target = train_images[index[i*batch_size:(i+1)*batch_size],:,:,:].float().to(device)
        MSE_loss = Bias_AE.AE_update(content, target)
    print('Epoch: %d, MSE_loss: %f'% (ep, MSE_loss))
print('Content Bias Autoencoder is trained!')

Bias_DE = Sty_Trainer()
#Train the Bias Decoder
print('----------------------------------------')
print('Start training Bias Decoder...')
Bias_DE.to(device)
for ep in range(epoch):
    index = torch.randperm(train_num_samples)
    for i in range(num_itr):
        const_style = train_Bias_style[index[i*batch_size:(i+1)*batch_size],:].float().to(device)
        target = train_images[index[i*batch_size:(i+1)*batch_size],:,:,:].float().to(device)
        MSE_loss = Bias_DE.DE_update(const_style, target)
    print('Epoch: %d, MSE_loss: %f'% (ep, MSE_loss))
print('Bias Decoder is trained!')


Style_DE = Sty_Trainer()
#Train the Style Decoder
print('----------------------------------------')
print('Start training Style Decoder...')
Style_DE.to(device)
loss = []
for ep in range(epoch):
    index = torch.randperm(train_num_samples)
    for i in range(num_itr):
        style = train_style[index[i*batch_size:(i+1)*batch_size]].float().to(device)
        target = train_images[index[i*batch_size:(i+1)*batch_size],:,:,:].float().to(device)
        MSE_loss = Style_DE.DE_update(style, target)
        loss.append(MSE_loss)
    print('Epoch: %d, MSE_loss: %f'% (ep, MSE_loss))
print('Style Decoder is trained!')


#IoB Test
#calculate the test MSE losses
with torch.no_grad():
    batch_size = 1
    num_itr = int(test_num_samples / batch_size)
    print('Start testing Content Autoencoder...')
    Content_AE.to(device)
    mse_cont = 0
    for i in range(num_itr):
        content = test_content[i * batch_size:(i + 1) * batch_size, :, :, :].float().to(device)
        target = test_images[i * batch_size:(i + 1) * batch_size, :, :, :].float().to(device)
        MSE_loss = Content_AE.test(content, target)
        mse_cont += MSE_loss
    mse_cont /= test_num_samples
    print('Content Autoencoder MSE_loss: %f' % (mse_cont))
    print('Content Autoencoder is tested!')

    print('Start testing Content Bias Autoencoder...')
    Bias_AE.to(device)
    mse_cont_bias = 0
    for i in range(num_itr):
        content = test_Bias_content[i * batch_size:(i + 1) * batch_size, :, :, :].float().to(device)
        target = test_images[i * batch_size:(i + 1) * batch_size, :, :, :].float().to(device)
        MSE_loss = Bias_AE.test(content, target)
        mse_cont_bias += MSE_loss
    mse_cont_bias /= test_num_samples
    print('Content Bias Autoencoder MSE_loss: %f' % (mse_cont_bias))
    print('Content Bias Autoencoder is tested!')

    print('Start testing Style Decoder...')
    Style_DE.to(device)
    mse_sty = 0
    for i in range(num_itr):
        style = test_style[i * batch_size:(i + 1) * batch_size, :].float().to(device)
        target = test_images[i * batch_size:(i + 1) * batch_size, :, :, :].float().to(device)
        MSE_loss = Style_DE.test(style, target)
        mse_sty += MSE_loss
    mse_sty /= test_num_samples
    print('Style Decoder MSE_loss: %f' % (mse_sty))
    print('Style Decoder is tested!')

    print('Start testing Bias Decoder...')
    Bias_DE.to(device)
    mse_sty_bias = 0
    for i in range(num_itr):
        const_style = test_Bias_style[i * batch_size:(i + 1) * batch_size, :].float().to(device)
        target = test_images[i * batch_size:(i + 1) * batch_size, :, :, :].float().to(device)
        MSE_loss = Bias_DE.test(const_style, target)
        mse_sty_bias += MSE_loss
    mse_sty_bias /= test_num_samples
    print('Bias Decoder MSE_loss: %f' % (mse_sty_bias))
    print('Bias Decoder is tested!')

IoBc = mse_cont_bias / mse_cont
IoBs = mse_sty_bias / mse_sty
print('IoBc is %f, IoBs is %f' % (IoBc, IoBs))

#Save results
print('IoBc is %f, IoBs is %f' % (IoBc, IoBs))
file = open(result_directory, 'a')
file.write('\nIoB metric for ' + dir_root + ':\n')
file.write('MSE Content Bias: ' + str(mse_cont_bias) + '\n')
file.write('MSE Content: ' + str(mse_cont) + '\n')
file.write('MSE Style Bias: ' + str(mse_sty_bias) + '\n')
file.write('MSE Style: ' + str(mse_sty) + '\n')
file.write('IoBc: ' + str(IoBc) + '\n')
file.write('IoBs: ' + str(IoBs) + '\n')
file.close()