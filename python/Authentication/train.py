# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
import argparse
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from gmm import GaussianMixture
from density import GaussianDensityTorch
from dataset import My_PCG, Repeat
from cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model import ProjectionNet as Net
from eval import eval_model
from utils import str2bool

def run_training(datapath,
                 test_path,
                 model_dir="models",
                 model_type="model",
                 pos_class="good",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 cutpate_type=CutPasteNormal,
                 device = "cuda",
                 workers=8,
                 size = 256):
    
    torch.multiprocessing.freeze_support()
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used
    temperature = 0.2

    weight_decay = 0.00003
    momentum = 0.9
    #TODO: use f strings also for the date LOL
    model_name = model_type



    Path(model_dir).mkdir(exist_ok=True, parents=True)

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    # after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                 std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    # train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpate_type(transform = after_cutpaste_transform))
    # train_transform.transforms.append(transforms.ToTensor())

    train_data = My_PCG(datapath, pos_class=pos_class, transform = train_transform)
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # create Model:
    head_layers = [512]*head_layer+[128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    # model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model = Net(num_classes)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        #scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    step = 0
    num_batches = len(dataloader)
    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out
    dataloader_inf =  get_data_inf()
    # From paper: "Note that, unlike conventional definition for an epoch,
    #              we define 256 parameter update steps as one epoch.
    max_roc_auc = 0
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            if i == freeze_resnet:
                model.unfreeze()
            xs = [x.to(device) for x in data]
            optimizer.zero_grad()
            xc = torch.cat(xs, axis=0)
            embeds, logits = model(xc)
            y = torch.arange(len(xs), device=device)
            y = y.repeat_interleave(xs[0].size(0))
            loss = loss_fn(logits, y)
            
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch)
            
            writer.add_scalar('loss', loss.item(), step)
            
    #         predicted = torch.argmax(ip,axis=0)
            predicted = torch.argmax(logits,axis=1)
    #         print(logits)
    #         print(predicted)
    #         print(y)
            accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
            writer.add_scalar('acc', accuracy, step)
            if scheduler is not None:
                writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

            writer.add_scalar('epoch', epoch, step)
            print("Epochs:[{}/{}]\tBatch[{}/{}]\tLoss:{:.6f}\ttrain_acc:{:.6f}".format(epoch, epochs, i, num_batches, loss, accuracy))

            if test_epochs > 0 and i % test_epochs == 0:
                # run auc calculation
                #TODO: create dataset only once.
                #TODO: train predictor here or in the model class itself. Should not be in the eval part
                #TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
                # batch_embeds = torch.cat(batch_embeds)
                # print(batch_embeds.shape)
                model.eval()
                roc_auc= eval_model(model_name, datapath, test_path, device=device,pos_class=pos_class,
                                    save_plots=False,
                                    size=size,
                                    show_training_data=False,
                                    model=model,
                                    mode="test",
                                    density=GaussianDensityTorch())
                                    #train_embed=batch_embeds)
                print(roc_auc)
                model.train()
                writer.add_scalar('eval_auc', roc_auc, step)
                if roc_auc > max_roc_auc:
                    torch.save(model.state_dict(), model_dir / f"{model_name}.tch")
                    max_roc_auc = roc_auc
        
        model.eval()
        roc_auc= eval_model(model_name, datapath, test_path, device=device,pos_class=pos_class,
                            save_plots=False,
                            size=size,
                            show_training_data=False,
                            model=model,
                            mode="test",
                            density=GaussianDensityTorch())
                            #train_embed=batch_embeds)
        print(roc_auc)
        model.train()
        writer.add_scalar('eval_auc', roc_auc, step)
        if roc_auc > max_roc_auc:
            torch.save(model.state_dict(), model_dir / f"{model_name}.tch")
            max_roc_auc = roc_auc



parser = argparse.ArgumentParser('Train')
parser.add_argument("--trainpath", type=str)
parser.add_argument("--evalpath",  type=str)
parser.add_argument("--model_dir",  type=str, default="models/evaluate/cwt/")
parser.add_argument("-e", "--epochs", default=1, type=int,help="Epochs")
parser.add_argument("-t", "--eval_epochs", default=1, type=int)
parser.add_argument("-b", "--batch_size", default=16, type=int,help="Batch Size")
parser.add_argument("-l", "--lr", default=2e-4, type=float)
parser.add_argument("--optim", default="adam", type=str)
parser.add_argument("--pretrained", default=True, type=str2bool)
parser.add_argument("--freeze_resnet", default=20, type=int)
parser.add_argument("--head_layer", default=1, type=int)
parser.add_argument('--variant', default="scar", choices=['normal', 'scar', '3way', 'union'], help='cutpaste variant to use')
parser.add_argument('--cuda', default=True, type=str2bool, help='use cuda for training')
parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading")
parser.add_argument('--type', default="all", help='dataset users to train seperated by , (default: "all": train all users)')
parser.add_argument("--model_name", default="default_gau", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    for item in dir(args):
        if not item.startswith('__'):
            print(item+" : "+str(getattr(args, item)))

    variant_map = {'normal':CutPasteNormal, 'scar':CutPasteScar, '3way':CutPaste3Way, 'union':CutPasteUnion}
    variant = variant_map[args.variant]
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    
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

    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        for item in dir(args):
            if not item.startswith('__'):
                f.write(item+" : "+str(getattr(args, item)) + "\n")

    for pos_class in pos_classes:
        print(f"training {pos_class}")
        # print(args.evalpath+str(int(pos_class[-3:])))
        run_training(datapath=args.trainpath,
                     test_path=args.evalpath+str(int(pos_class[-3:])),
                     model_dir=Path(args.model_dir + pos_class),
                     model_type=args.model_name,
                     epochs=args.epochs,
                     pos_class=pos_class,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learninig_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=variant,
                     workers=args.workers)
