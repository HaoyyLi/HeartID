import logging
import os  # 引入logging模块
import h5py
import torch
import numpy as np
from torchvision import transforms, datasets

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_img_loader(config, cset):
    rootpath = config.dataset_path+"/"+cset+"set/"
    if cset == 'test':
        shuffle_ = False
    else:
        shuffle_ = True
    data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
                                    root=rootpath,
                                    transform=transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor()
                                    ])), 
                                batch_size=config.batch_size, 
                                shuffle=shuffle_, 
                                drop_last=False)
    return data_loader


def get_mat_loader(config, cset):
    rootpath = config.dataset_path+"/"+cset+"set/"
    filelist = os.listdir(rootpath)
    filelist.sort()
    y0 = 0
    X = []
    Y = []
    for file in filelist:
        data = h5py.File(rootpath+file)
        X_i = np.array(data["X"])
        X.append(X_i)
        Y.append(np.ones((X_i.shape[0],),dtype=np.int16)*y0)
        y0+=1
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    X = X[:,None,:,:]
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    dataset = torch.utils.data.TensorDataset(X, Y)
    
    if cset == 'test':
        shuffle_ = False
    else:
        shuffle_ = True

    data_Loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = config.batch_size,
        shuffle=shuffle_,
    )
    return data_Loader

if __name__ == '__main__':
    from config import Config
    cfg = Config()
    get_mat_loader(cfg, "test")