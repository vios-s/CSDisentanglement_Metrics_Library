## This script is an example of extracting the image and content tensors as well as the style vector
#  from a pre-train MUNIT model.
## This script is not ready to run, where a pre-trained model and its corresponding designs are not given.

from utils import get_config, get_all_data_loaders, __write_images
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--config', type=str, default='config.yaml', help="net configuration")
parser.add_argument('--output_dir', type=str, default='', help='output directory')
parser.add_argument('--checkpoint', type=str, default='checkpoints/gen_00120000.pt', help="checkpoint of autoencoders") #change here
parser.add_argument('--seed', type=int, default=14, help="random seed")
parser.add_argument('--num_style',type=int, default=150, help="number of styles to sample")
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

##################### Load experiment setting ###########################################
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu
config = get_config(opts.config)
style_dim = config['gen']['style_dim']
batch_size = config['batch_size']
output_dir = opts.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

########################## Load model #################################################
trainer = MUNIT_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()

encode_a = trainer.gen_a.encode # encoder A function
decode_a = trainer.gen_a.decode # decoder A function
encode_b = trainer.gen_b.encode # encoder B function
decode_b = trainer.gen_b.decode # decoder B function

########################## Load data #################################################
_, _, test_loader_a, test_loader_b = get_all_data_loaders(config)

if not os.path.exists(output_dir+'/'+'output_images'):
    os.makedirs(output_dir+'/'+'output_images')

############### Extract the tensors and vectors ######################################
# Note that the batch size is set as 300.
with torch.no_grad():
    for images in test_loader_a:
        if images.size(0) != batch_size:
            continue
        images = images.cuda().detach()
        # Start testing
        test_images_a = images.cpu().numpy()
        c_a, s_a = encode_a(images)
        c_a_np = c_a.cpu().numpy()
        s_a_np = s_a.squeeze().cpu().numpy()

    for images in test_loader_b:
        if images.size(0) != batch_size:
            continue
        images = images.cuda().detach()
        # Start testing
        test_images_b = images.cpu().numpy()
        c_b, s_b = encode_b(images)
        c_b_np = c_b.cpu().numpy()
        s_b_np = s_b.squeeze().cpu().numpy()

############### Save the tensors and vectors ########################################
    np.savez_compressed(output_dir+'/'+'content_test', np.concatenate((c_a_np, c_b_np), axis=0))
    np.savez_compressed(output_dir+'/'+'style_test', np.concatenate((s_a_np, s_b_np), axis=0))
    np.savez_compressed(output_dir+'/'+'images_test', np.concatenate((test_images_a, test_images_b), axis=0))