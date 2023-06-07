import os
import argparse
import torch
from torch import nn
from utils import *
from HeartNet import HN as Net
from tqdm import tqdm

def test_net(testLoader, model):
    acc = 0
    num = 0
    model.eval()
    for i, (x, target) in enumerate(testLoader):
        x = x.to(device).to(torch.float32)
        out = model(x)
        acc += torch.sum(torch.argmax(out.cpu().detach(),dim=1)==target).item()
        num += out.size()[0]
    return acc / num
def train_net(trainLoader, validLoader, model, Loss_func, optimizer, model_path, log_Path):
    logger = get_logger(log_Path)
    logger.info('start training!')
    MAX_ACC = 0
    for e in range(epochs):
        model.train()
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader)) # create a progress bar
        for i, (x, target) in loop:
            x = x.to(device).to(torch.float32)
            target = target.to(device).long()
            optimizer.zero_grad()
            out = model(x)
            loss = Loss_func(out, target)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_acc = torch.sum(torch.argmax(out.cpu().detach(),dim=1)==target.cpu()).item() / x.size()[0]
            # print("Epochs:[{}/{}]\tBatch[{}/{}]\tLoss:{:.6f}\ttrain_acc:{:.6f}".format(e, epochs, i, len(trainLoader), loss, train_acc))
            loop.set_description("Epoch {}/{} Batch: {}/{} Loss:{:.4f} train_acc:{:.2f}".format(e, epochs, int(i), int(len(trainLoader)), loss, train_acc*100))
        acc = test_net(validLoader, model)
        loop.set_postfix(loss=loss.data.item(), acc=acc)
        logger.info("Epochs:[{}/{}]\tvalid_acc:{:.6f}".format(e, epochs, acc))
        

        # 设置 early stop
        if acc > MAX_ACC:
            torch.save(model.state_dict(), model_path)
            MAX_ACC = acc
    return model

parser = argparse.ArgumentParser('Train')
parser.add_argument("--dataset_path", type=str)
parser.add_argument("-e", "--epochs", default=50, type=int,help="Epochs")
parser.add_argument("-b", "--batch_size", default=16, type=int,help="Batch Size")
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-l", "--lr", default=2e-4, type=float)
parser.add_argument("--modelname", default="HN_604", type=str)

def __init__():
    if not os.path.exists("model/"):
        os.makedirs("model/")
    if not os.path.exists("log/"):
        os.makedirs("log/")

if __name__=='__main__':
    __init__()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_torch(args.seed)
    epochs = args.epochs

    model_path = "model/"+"/"+args.modelname+".pkl"
    log_Path = "log/"+"/"+args.modelname+".log"

    trainLoader = get_img_loader(args, "train")
    validLoader = get_img_loader(args, "valid")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=[0.5, 0.99])
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    Loss_func = nn.CrossEntropyLoss()
    model = train_net(trainLoader, validLoader, model, Loss_func, optimizer, model_path, log_Path)

