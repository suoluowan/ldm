import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class dpSingleBase(Dataset):
    def __init__(self,
                 txt_file1,
                 data_root1,
                 iuv_root1,
                 txt_file2,
                 data_root2,
                 iuv_root2,
                 txt_file3=None,
                 data_root3="",
                 iuv_root3="",
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
        self.data_paths3 = txt_file3
        self.data_root3 = data_root3
        self.iuv_root3 = iuv_root3
        with open(self.data_paths1, "r") as f:
            self.image_paths1 = f.read().splitlines()
        with open(self.data_paths2, "r") as f:
            self.image_paths2 = f.read().splitlines()
        if self.data_paths3 is not None:
            with open(self.data_paths3, "r") as f:
                self.image_paths3 = f.read().splitlines()
        else:
            self.image_paths3 = []
        self._length = len(self.image_paths1) + len(self.image_paths2) + len(self.image_paths3)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths1] + [l for l in self.image_paths2] + [l for l in self.image_paths3],
            "file_path_": [os.path.join(self.data_root1, l)
                           for l in self.image_paths1] + 
                           [os.path.join(self.data_root2, l)
                           for l in self.image_paths2] + 
                           [os.path.join(self.data_root3, l)
                           for l in self.image_paths3],
            "iuv_file_path_": [os.path.join(self.iuv_root1, l.split('.')[0]+'.exr')
                           for l in self.image_paths1] + 
                           [os.path.join(self.iuv_root2, l.split('.')[0]+'.exr')
                           for l in self.image_paths2] + 
                           [os.path.join(self.iuv_root3, l.split('.')[0]+'.exr')
                           for l in self.image_paths3],
        }

        self.size = size
        self.interpolation_iuv = {"linear": cv2.INTER_LINEAR,
                                "bilinear": cv2.INTER_LINEAR,
                                "bicubic": cv2.INTER_CUBIC,
                                "lanczos": cv2.INTER_LANCZOS4,
                                }[interpolation]
        self.interpolation_img = {"linear": PIL.Image.LINEAR,
                                "bilinear": PIL.Image.BILINEAR,
                                "bicubic": PIL.Image.BICUBIC,
                                "lanczos": PIL.Image.LANCZOS,
                                }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        iuv = cv2.imread(example["iuv_file_path_"], cv2.IMREAD_UNCHANGED)
        iuv[:,:,0] = np.clip(iuv[:,:,0]/24. *255., 0, 255)
        iuv[:,:,1] = np.clip(iuv[:,:,1] *255., 0, 255)
        iuv[:,:,2] = np.clip(iuv[:,:,2] *255., 0, 255)
        if self.size is not None:
            iuv_image = cv2.resize(iuv, (self.size, self.size), interpolation=self.interpolation_iuv)
        example["iuv"] = (iuv_image / 127.5 - 1.0).astype(np.float32)
        
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation_img)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example


class dpSingleTrain(dpSingleBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file1="/share/home/liuqiong/datasets/coco/annotations/densepose_train2014_single_filted.txt", data_root1="/share/home/liuqiong/datasets/coco/dpsingleMask/train2014", iuv_root1="/share/home/liuqiong/datasets/coco/IUV_EXR/dpsingleMask/train2014", txt_file2="/share/home/liuqiong/datasets/coco/annotations/densepose_valminusminival2014_single.txt", data_root2="/share/home/liuqiong/datasets/coco/dpsingleMask/val2014", iuv_root2="/share/home/liuqiong/datasets/coco/IUV_EXR/dpsingleMask/val2014", txt_file3="/share/home/liuqiong/datasets/ultrapose/annotations/densepose_train2014.txt", iuv_root3="/share/home/liuqiong/datasets/ultrapose/IUV_EXR/singleMask/train2014", data_root3="/share/home/liuqiong/datasets/ultrapose/singleMask/train2014",**kwargs)


class dpSingleValidation(dpSingleBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file1="/share/home/liuqiong/datasets/coco/annotations/densepose_minival2014_single.txt", data_root1="/share/home/liuqiong/datasets/coco/dpsingleMask/val2014", iuv_root1="/share/home/liuqiong/datasets/coco/IUV_EXR/dpsingleMask/val2014", txt_file2="/share/home/liuqiong/datasets/ultrapose/annotations/densepose_valminusminival2014.txt", iuv_root2="/share/home/liuqiong/datasets/ultrapose/IUV_EXR/singleMask/val2014", data_root2="/share/home/liuqiong/datasets/ultrapose/singleMask/val2014", flip_p=flip_p, **kwargs)


