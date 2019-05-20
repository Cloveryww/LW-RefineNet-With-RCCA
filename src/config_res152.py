import numpy as np

# DATASET PARAMETERS
#TRAIN_DIR = '/datasets/nyud/'
DATASET='VOC'
TRAIN_DIR = '/home/xiejinluo/data/PASCAL_VOC/VOCdevkit_aug/VOC2012/'
VAL_DIR = TRAIN_DIR
#TRAIN_LIST = ['./data/train.nyu'] * 3
#VAL_LIST = ['./data/val.nyu'] * 3
TRAIN_LIST = ['/home/xiejinluo/data/PASCAL_VOC/VOCdevkit_aug/VOC2012/trainval2.txt'] * 3
VAL_LIST = ['/home/xiejinluo/data/PASCAL_VOC/VOCdevkit_aug/VOC2012/val2.txt'] * 3

TEACHER_URL = ''
TEACHER_LAMBD = 1.0

SHORTER_SIDE = [350] * 3
CROP_SIZE = [500] * 3
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD
BATCH_SIZE = [6] * 3
#BATCH_SIZE = [2] * 3
NUM_WORKERS = 16
#NUM_CLASSES = [40] * 3
NUM_CLASSES = [21] * 3
LOW_SCALE = [0.5] * 3
HIGH_SCALE = [2.0] * 3
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '50'
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 3
#NUM_SEGM_EPOCHS = [0,50,50]
NUM_SEGM_EPOCHS = [0,0,50]
#NUM_SEGM_EPOCHS = [50,200,100]
PRINT_EVERY = 10
RANDOM_SEED = 42
SNAPSHOT_DIR = './ckpt_res152/'
CKPT_PATH = './ckpt_res152/checkpoint.pth.tar'
#VAL_EVERY = [1] * 3 # how often to record validation scores
VAL_EVERY = [5] * 3 # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
#LR_ENC = [1e-3, 2.5e-4, 1.25e-4]  # TO FREEZE, PUT 0
LR_DEC = [5e-3, 1e-3, 5e-4]
#LR_ENC = [1e-3, 5e-4, 2.5e-4]  # TO FREEZE, PUT 0
#LR_DEC = [5e-3, 2.5e-3, 1e-3]
MOM_ENC = [0.9] * 3 # TO FREEZE, PUT 0
MOM_DEC = [0.9] * 3
WD_ENC = [1e-5] * 3 # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 3
OPTIM_DEC = 'sgd'
