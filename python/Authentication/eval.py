import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, roc_curve, classification_report
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import My_PCG
from cutpaste import CutPaste
from model import ProjectionNet as Net
import matplotlib.pyplot as plt
from pathlib import Path
from cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
import numpy as np
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
from gmm import GaussianMixture
import pandas as pd
import scipy.io as sio
from utils_for_class import saveclass
from utils import str2bool
import os

test_data_eval = None
test_transform = None
cached_type = None

def get_train_embeds(datapath, model, device, pos_class="good"):
    # train data / train kde
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ToTensor())
    test_data = My_PCG(datapath, transform=train_transform, pos_class=pos_class, mode="train")

    dataloader_train = DataLoader(test_data, batch_size=64,
                            shuffle=False, num_workers=0)
    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, logit = model(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

def eval_model(modelname, datapath, testpath, device="cpu", pos_class="good", save_plots=False, size=256, show_training_data=True, model=None, train_embed=None, head_layer=8, density=GaussianDensityTorch(), mode="test"):

    transform = transforms.Compose([])
    transform.transforms.append(transforms.ToTensor())
    test_data_eval = My_PCG(testpath, transform=transform, pos_class=pos_class, mode=mode)

    dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                    shuffle=False, num_workers=0)

    # create model
    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512]*head_layer+[128]
        print(head_layers)
        weights = torch.load(modelname)
        classes = weights["out.weight"].shape[0]
        # model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        # model, _ = Net()
        model = Net(classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    #get embeddings for test data
    labels = []
    embeds = []
    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x.to(device))

            # save 
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = torch.cat(labels)
    embeds = torch.cat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(datapath, model, device, pos_class)

    # norm embeds
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    #create eval plot dir
    if save_plots:
        eval_dir = Path("eval") / modelname
        eval_dir.mkdir(parents=True, exist_ok=True)
        # plot tsne
        # also show some of the training data
        show_training_data = False
        if show_training_data:
            #augmentation setting
            # TODO: do all of this in a separate function that we can call in training and evaluation.
            #       very ugly to just copy the code lol
            min_scale = 0.5

            # create Training Dataset and Dataloader
            after_cutpaste_transform = transforms.Compose([])
            after_cutpaste_transform.transforms.append(transforms.ToTensor())
            after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]))

            train_transform = transforms.Compose([])
            #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            #train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            train_transform.transforms.append(CutPaste(transform=after_cutpaste_transform))
            # train_transform.transforms.append(transforms.ToTensor())

            train_data = My_PCG(datapath)
            dataloader_train = DataLoader(train_data, batch_size=32,
                        shuffle=True, num_workers=8, collate_fn=cut_paste_collate_fn,
                        persistent_workers=True)
            # inference training data
            train_labels = []
            train_embeds = []
            with torch.no_grad():
                for x1, x2 in dataloader_train:
                    x = torch.cat([x1,x2], axis=0)
                    embed, logit = model(x.to(device))

                    # generate labels:
                    y = torch.tensor([0, 1])
                    y = y.repeat_interleave(x1.size(0))

                    # save 
                    train_embeds.append(embed.cpu())
                    train_labels.append(y)
                    # only less data
                    break
            train_labels = torch.cat(train_labels)
            train_embeds = torch.cat(train_embeds)

            # for tsne we encode training data as 2, and augmentet data as 3
            tsne_labels = torch.cat([labels, train_labels + 2])
            tsne_embeds = torch.cat([embeds, train_embeds])
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")

    
    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    # distances = density.predict(embeds)
    distances = density.distance(embeds)
    #TODO: set threshold on mahalanobis distances and use "real" probabilities

    fpr, tpr, thresholds = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)
    

    if __name__=='__main__':
        eval_dir = Path("eval") / modelname
        eval_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(fpr, tpr, eval_dir, modelname=modelname, save_plots=save_plots)
        fnr = 1-tpr
        eer_idx = np.abs(fpr-fnr).argmin()
        Th = thresholds[eer_idx]
        precision, recall, f1, _  = precision_recall_fscore_support(labels, distances>Th, pos_label=1, average="weighted")
        accuracy = accuracy_score(labels, distances>Th)
        print("precision:{:.6f}\trecall:{:.6f}\tf1score:{:.6f}\taccuracy:{:.6f}\tAUC:{:.6f}\tEER:{:.6f}".format(precision, recall, f1, accuracy, roc_auc,(fnr[eer_idx]+fpr[eer_idx])/2))
        reports = classification_report(labels, distances>Th, labels=[0, 1], target_names=["illegal", "legal"], output_dict=True)
        reports = pd.DataFrame(reports).transpose()
        reports.loc["AUC"]=roc_auc
        reports.loc["EER"]=(fnr[eer_idx]+fpr[eer_idx])/2
        print(reports)
        reports.to_csv(eval_dir / "reports.csv", index=True)
        sio.savemat(eval_dir / "TAR_FAR.mat",{"TAR":tpr, "FAR":fpr})
        if not os.path.exists(modelname[:-4]):
            os.mkdir(modelname[:-4])
        saveclass(modelname[:-4]+'/GaussianDensityTorch', density)
        sio.savemat(modelname[:-4]+'/GaussianDensityTorch.mat', {'Thr':thresholds, "TAR":tpr, "FAR":fpr, "EER_Th":Th})

    return roc_auc
    

def plot_roc(fpr, tpr, filename, modelname="", save_plots=False):
    #plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename / "roc_plot.png")
        plt.close()

        plt.figure()
        lw = 2
        plt.plot(fpr, 1-tpr, color='darkorange',
                lw=lw, label='EER curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename / "eer_plot.png")
        plt.close()

def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:,0], tsne_results[:,1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()

parser = argparse.ArgumentParser('Train')
parser.add_argument("--trainpath", type=str)
parser.add_argument("--testpath",  type=str)
parser.add_argument("--model_dir",  type=str, default="models/evaluate/cwt/")
parser.add_argument("-b", "--batch_size", default=16, type=int,help="Batch Size")
parser.add_argument("--optim", default="adam", type=str)
parser.add_argument("--head_layer", default=1, type=int)
parser.add_argument('--variant', default="scar", choices=['normal', 'scar', '3way', 'union'], help='cutpaste variant to use')
parser.add_argument('--cuda', default=True, type=str2bool, help='use cuda for training')
parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading")
parser.add_argument('--type', default="all", help='dataset users to train seperated by , (default: "all": train all users)')
parser.add_argument("--model_name", default="default_gau", type=str)
parser.add_argument("--save_plots", default=False, type=bool)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    device = "cuda" if args.cuda else "cpu"
    
    if args.type=="all":
        pos_classes = ["USERS001",
                       "USERS002",
                       "USERS003",
                       "USERS004",
                       "USERS005",
                       "USERS006",
                       "USERS007",
                       "USERS004",
                       "USERS005",
                       "USERS006",
                       "USERS007",
                       "USERS008",
                       "USERS009",
                       "USERS010",
                       "USERS011",
                       "USERS012",
                       "USERS013",
                       "USERS014"]
    else:
        pos_classes = [args.type]

    obj = defaultdict(list)
    density = GaussianDensityTorch()


    for pos_class in pos_classes:
        model_name = args.model_dir+"/"+pos_class+"/"+args.model_name+".tch"
        roc_auc = eval_model(model_name, args.trainpath, args.testpath+str(int(pos_class[-3:])), device=device, pos_class=pos_class, save_plots=args.save_plots, head_layer=args.head_layer, density=args.density, mode="test")
        print(f"{pos_class} AUC: {roc_auc}")
        obj["pos_classes"].append(pos_class)
        obj["roc_auc"].append(roc_auc)
    
        # save pandas dataframe
        eval_dir = Path("eval") / args.model_dir
        eval_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(obj)
        df.to_csv(str(eval_dir) + "_perf.csv")
