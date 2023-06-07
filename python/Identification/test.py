# %%
import argparse
import torch
from HeartNet import HN as Net
from utils import *
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser('Train')
parser.add_argument("--dataset_path", type=str)
parser.add_argument("-e", "--epochs", default=50, type=int,help="Epochs")
parser.add_argument("-b", "--batch_size", default=16, type=int,help="Batch Size")
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-l", "--lr", default=2e-4, type=float)
parser.add_argument("--modelname", default="HN_604", type=str)

def __init__():
    if not os.path.exists("model/"):
        raise KeyError("error: not find model path")
    if not os.path.exists("result/"):
        os.makedirs("result/") 

if __name__=='__main__':
    __init__()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "model/"+"/"+args.modelname+".pkl"
    result_path = "result/"+"/"+args.modelname
    testLoader = get_img_loader(args, "test")
    model = Net().to(device)
    acc = 0
    num = 0
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    prev = []
    label = []
    model.eval()
    for _, (x, target) in enumerate(testLoader):
        x = x.to(device).to(torch.float32)
        out = model(x)
        out = torch.argmax(out.cpu().detach(),dim=1)
        prev.append(out.numpy())
        label.append(target.numpy().astype(np.int16))
    prev = np.concatenate(prev)
    label = np.concatenate(label)

    sns.set_context(context="paper")
    # 混淆矩阵 矩阵  行：分类 列：分类
    labels=["U1","U2","U3","U4","U5","U6","U7","U8","U9","U10","U11","U12","U13","U14"]
    # labels=["U4","U5","U6","U7","U8","U9","U10","U11"]
    # labels=["U2","U3","U4","U5","U6","U7","U8","U10","U11","U13","U14"]
    cm = sm.confusion_matrix(label, prev, labels=list(np.arange(len(labels))))
    cm = cm / cm.sum(axis=1).reshape(-1,1)
    cm = pd.DataFrame(cm,columns=labels,index=labels)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm,cmap="GnBu",annot=True)
    plt.xlabel("y_Prev")
    plt.ylabel("y_True")
    plt.savefig(result_path+'.jpg')
    plt.show()
    print("---------------混淆矩阵\n", cm)

    cp = sm.classification_report(label, prev, labels=list(np.arange(len(labels))), target_names=labels,output_dict=True)
    df = pd.DataFrame(cp).transpose()
    print(df)
    df.to_csv(result_path+".csv", index=True)

