import time
import sys
import os
sys.path.append('../')

from models.mobilenet import mbv2
from models.resnet import rf_lw_model
from models.mobilenet import rf_lw_mbv2_model

from utils.helpers import prepare_img
import cv2
import numpy as np
import torch

from PIL import Image

cmap = np.load('../utils/cmap.npy')
cmap = cmap[:21,:]
has_cuda = torch.cuda.is_available()
n_classes = 21

test_list = '/home/xiejinluo/data/PASCAL_VOC/VOCdevkit_aug/VOC2012/speed_test/test.txt'
imgs_dir = '/home/xiejinluo/data/PASCAL_VOC/VOCdevkit_aug/VOC2012/speed_test/JPEGImages/'
gt_dir = '/home/xiejinluo/data/PASCAL_VOC/VOCdevkit/VOC2012/SegmentationClass/'
# get one model
#mbv2
model_url = '/home/xiejinluo/yww/light-weight-refinenet/ckpt_coco_mbv2/coco_mbv2_bs6_1n/checkpoint330epochwith_coco.pth.tar'

#res50
#model_url = '/home/xiejinluo/yww/light-weight-refinenet/ckpt_coco/coco_res50_6bs_1n/checkpoint323.pth.tar'
#net = rf_lw_model(n_classes, "res50", model_url)
net = rf_lw_mbv2_model(n_classes, "mbv2", model_url)
if has_cuda:
    print("has cuda")
    net = torch.nn.DataParallel(net).cuda()
else:
    print("has not cuda")
net = net.eval()
cnt = 0
time_list = []
with torch.no_grad():
    for img_name in open(test_list,'r').readlines():
        img_name = img_name.strip()
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path): 
            print("Can not find "+img_path)
            continue
        img = np.array(Image.open(img_path))
        orig_size = img.shape[:2][::-1]
        t1 = time.time()
        img_inp = torch.autograd.Variable(torch.tensor(prepare_img(img).transpose(2, 0, 1)[None])).float()
        if has_cuda:
            img_inp = img_inp.cuda()
        segm = net(img_inp)[0, :n_classes].data.cpu().numpy().transpose(1, 2, 0)
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
        segm2 = segm.argmax(axis=2).astype(np.uint8)
        t2 = time.time()
        time_list.append(t2-t1)
        print("cnt: ",cnt," time: ",t2-t1)
        #segm = cmap[segm2]
        #segm=cv2.cvtColor(segm, cv2.COLOR_BGR2RGB)
        #print (segm.shape)
        #cv2.imwrite('tests/'+img_name+'_grey.png',segm2)
        cv2.imwrite('speed_out/'+img_name[:-4]+'.png',segm2)
        #os.system("cp "+img_path+" tests/")
        #gt_path = os.path.join(gt_dir, img_name+'.png')
        #os.system("cp "+gt_path+" tests/"+img_name+'_gt.png')
	#cv2.imwrite('seg_color.png',segm2)
        #tee = np.array(Image.open(gt_path))
        #print(tee.shape)
        #tee = cmap[tee]
	#cv2.imwrite('tests/'+img_name+'gt_color.png',tee)
time_list = time_list[1:]
ave = np.mean(time_list)
std = np.std(time_list,ddof=1)
print("mean = ",ave,"std = ",std,"fps = ",1.0/ave)

