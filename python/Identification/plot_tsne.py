# %%
import torch
# from model import MobileNetV2 as Net
from HeartNet_tsne import HN as Net
# from model import ResNet as Net
# from HeartNet2 import MViT2 as Net
# from model import MY_CNN as Net
from utils import *
from config import CONFIG
# %%
cfg = CONFIG()
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
testLoader = get_img_loader(cfg, "test")
model = Net().to(device)
acc = 0
num = 0
model.load_state_dict(torch.load(cfg.model_path, map_location=torch.device(device)))
x_test = []
prev = []
label = []
model.eval()
for _, (x, target) in enumerate(testLoader):
    x = x.to(device).to(torch.float32)
    out = model(x)
    x_test.append(x.cpu().detach().numpy())
    prev.append(out.cpu().detach().numpy())
    label.append(target.numpy().astype(np.int16))
x_test = np.concatenate(x_test)
x_test = x_test.reshape(x_test.shape[0],-1)
prev = np.concatenate(prev)
label = np.concatenate(label)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

tsne = TSNE(n_components=2, init='pca', random_state=0)
x_result = tsne.fit_transform(x_test)
tsne = TSNE(n_components=2, init='pca', random_state=0)
p_result = tsne.fit_transform(prev)

cmap = mpl.cm.get_cmap("jet", 14)
colormap = cmap(np.linspace(0, 1, 14))

fig, ax = plt.subplots(1)
ax.scatter(x_result[:,0], x_result[:,1], color=[colormap[l] for l in label])
plt.show()
plt.savefig("./tsne/cwt.jpg")
plt.close()
fig, ax = plt.subplots(1)
ax.scatter(p_result[:,0], p_result[:,1], color=[colormap[l] for l in label])
plt.show()
plt.savefig("./tsne/net.jpg")
plt.close()
import scipy.io as sio
sio.savemat("./tsne/tsne.mat", {"x_cwt": x_result, "x_net": p_result, "label": label})