import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class dpSingleBase(Dataset):
    def __init__(self,
                 txt_file1,
                 data_root1,
                 iuv_root1,
                 txt_file2=None,
                 data_root2="",
                 iuv_root2="",
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths1 = txt_file1
        self.data_root1 = data_root1
        self.iuv_root1 = iuv_root1
        self.data_paths2 = txt_file2
        self.data_root2 = data_root2
        self.iuv_root2 = iuv_root2
        with open(self.data_paths1, "r") as f:
            self.image_paths1 = f.read().splitlines()
        if self.data_paths2 is not None:
            with open(self.data_paths2, "r") as f:
                self.image_paths2 = f.read().splitlines()
        else:
            self.image_paths2 = []
        self._length = len(self.image_paths1) + len(self.image_paths2)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths1] + [l for l in self.image_paths2],
            "file_path_": [os.path.join(self.data_root1, l)
                           for l in self.image_paths1] + 
                           [os.path.join(self.data_root2, l)
                           for l in self.image_paths2],
            "iuv_file_path_": [os.path.join(self.iuv_root1, l)
                           for l in self.image_paths1] + 
                           [os.path.join(self.iuv_root2, l)
                           for l in self.image_paths2],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        iuv = Image.open(example["iuv_file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if not iuv.mode == "RGB":    
            iuv = iuv.convert("RGB")

        img = np.array(iuv).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        scale = min(self.size/h, self.size/w)
        nw = int(w * scale)
        nh = int(h * scale)
        # img = img[(h - crop) // 2:(h + crop) // 2,
        #       (w - crop) // 2:(w + crop) // 2]

        iuv = Image.fromarray(img)
        if self.size is not None:
            iuv = iuv.resize((self.size, self.size), resample=self.interpolation)
            # new_iuv = Image.new('RGB', (self.size,self.size), (0, 0, 0))
            # new_iuv.paste(iuv, ((self.size - nw) // 2, (self.size - nh) // 2))

        iuv = self.flip(iuv)
        iuv = np.array(iuv).astype(np.uint8)
        example["iuv"] = (iuv / 127.5 - 1.0).astype(np.float32)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        scale = min(self.size/w, self.size/h)
        nw = int(w * scale)
        nh = int(h * scale)
        # img = img[(h - crop) // 2:(h + crop) // 2,
        #       (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            # new_image = Image.new('RGB', (self.size,self.size), (0, 0, 0))
            # new_image.paste(image, ((self.size - nw) // 2, (self.size - nh) // 2))

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example


class dpSingleTrain(dpSingleBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file1="/share/home/liuqiong/datasets/coco/annotations/densepose_train2014_single.txt", data_root1="/share/home/liuqiong/datasets/coco/dpsingleMask/train2014", iuv_root1="/share/home/liuqiong/datasets/coco/dpsingleMask/IUV/train2014", txt_file2="/share/home/liuqiong/datasets/coco/annotations/densepose_valminusminival2014_single.txt", data_root2="/share/home/liuqiong/datasets/coco/dpsingleMask/val2014", iuv_root2="/share/home/liuqiong/datasets/coco/dpsingleMask/IUV/val2014", **kwargs)


class dpSingleValidation(dpSingleBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file1="/share/home/liuqiong/datasets/coco/annotations/densepose_minival2014_single.txt", data_root1="/share/home/liuqiong/datasets/coco/dpsingleMask/val2014", iuv_root1="/share/home/liuqiong/datasets/coco/dpsingleMask/IUV/val2014", flip_p=flip_p, **kwargs)


