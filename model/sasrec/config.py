

import torch

class AttributeDict(dict):
    def __init__(self):
        self.__dict__ = self

class ConfigTree:
    def __init__(self):
        self.SYSTEM = AttributeDict()
        self.PATH = AttributeDict()
        self.DATASET = AttributeDict()
        self.TRAIN = AttributeDict()
        self.MODEL = AttributeDict()

        self.KD = AttributeDict()

def get_config():
    config = ConfigTree()
    config.SYSTEM.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    config.PATH.SAVEDIR = './checkpoint'
    config.PATH.ROOT = '/opt/ml/input/data/train/images'
    config.PATH.RESUME_1 = '/opt/ml/code/checkpoint/DenseNet_9'
    config.PATH.TEST_1 = '/opt/ml/code/checkpoint/DenseNet_10'
    

    config.DATASET.RATIO = 0.15

    config.TRAIN.DROPOUT = 0.5                  
    config.TRAIN.EPOCH = 50 #args.epochs
    config.TRAIN.BATCH_SIZE = 64 #args.batch_size
    config.TRAIN.NUM_WORKERS = 5
    config.TRAIN.BASE_LR = 1e-4 #args.lr 
    config.TRAIN.PERIOD = 1
    config.TRAIN.RESUME = False

    config.TRAIN.NUM_BLOCKS = 2
    config.TRAIN.NUM_HEADS = 1
    config.TRAIN.L2_EMD = 0.0
    config.TRAIN.MAXLEN= 200
    config.MODEL.OPTIM = 'Adam'
    config.MODEL.HIDDEN = 512
    

    return config



