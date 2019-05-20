import six
import sys
sys.path.append('../../')

from models.mobilenet import mbv2
from models.resnet import rf_lw50, rf_lw101, rf_lw152

from utils.helpers import prepare_img

%matplotlib inline

import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

cmap = np.load('../../utils/cmap.npy')
has_cuda = torch.cuda.is_available()
img_dir = '../imgs/VOC/'
imgs = glob.glob('{}*.jpg'.format(img_dir))
n_classes = 21

# Initialise models
model_inits = { 
    'rf_lw50_voc'   : rf_lw50, # key / constructor
    'rf_lw101_voc'  : rf_lw101,
    'rf_lw152_voc'  : rf_lw152,
    'rf_lwmbv2_voc': mbv2
    }

models = dict()
for key,fun in six.iteritems(model_inits):
    net = fun(n_classes, pretrained=True).eval()
    if has_cuda:
        net = net.cuda()
    models[key] = net

# Figure 2 from the paper
n_cols = len(models) + 2 # 1 - for image, 1 - for GT
n_rows = len(imgs)

plt.figure(figsize=(16, 12))
idx = 1

with torch.no_grad():
    for img_path in imgs:
        img = np.array(Image.open(img_path))
        msk = cmap[np.array(Image.open(img_path.replace('jpg', 'png')))]
        orig_size = img.shape[:2][::-1]
        
        img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
        if has_cuda:
            img_inp = img_inp.cuda()
        
        plt.subplot(n_rows, n_cols, idx)
        plt.imshow(img)
        plt.title('img')
        plt.axis('off')
        idx += 1
        
        plt.subplot(n_rows, n_cols, idx)
        plt.imshow(msk)
        plt.title('gt')
        plt.axis('off')
        idx += 1
        
        for mname, mnet in six.iteritems(models):
            segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
            segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
            segm = cmap[segm.argmax(axis=2).astype(np.uint8)]
            
            plt.subplot(n_rows, n_cols, idx)
            plt.imshow(segm)
            plt.title(mname)
            plt.axis('off')
            idx += 1
