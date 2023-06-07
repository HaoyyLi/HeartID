from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
from torchvision import transforms

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class My_PCG(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform, pos_class="good", mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.size = 256
        self.pos_class = pos_class
        
        # find test images
        if self.mode == "train":
            self.image_names = list((self.root_dir / "trainset" / pos_class).glob("*.png"))
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((256,256)).convert("RGB"))(file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")
        elif self.mode == "valid":
            self.image_names = list((self.root_dir / "validset").glob(str(Path("*") / "*.png")))
        else:
            #test mode
            self.image_names = list((self.root_dir / "testset").glob(str(Path("*") / "*.png")))
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != self.pos_class
