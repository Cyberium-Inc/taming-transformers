import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import glob
from functools import partial
import time
import albumentations
from omegaconf import OmegaConf
import cv2
import json
from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.base import ImagePaths


class Base(Dataset):
    def __init__(self,
                 txt_file,
                 process_images=True,
                 state=None,
                 config=None
                 ):

        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get(
            "keep_orig_class_label", False)
        
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
       

        self.process_images = process_images
        self.txt_file = txt_file
        self._load()


    def _load(self):
        with open(self.txt_file, 'r') as fp:
            data = json.load(fp)
        
        with open(self.txt_file.replace("jsonlinessanskrit-captions","instances"), 'r') as fp:
            instances = json.load(fp)
        """
        #Target to match this format of imagenet
        >>> labels
        {'relpath': array(['n01440764/n01440764_10026.JPEG', 'n01440764/n01440764_10027.JPEG',
            'n01440764/n01440764_10029.JPEG', ...,
            'n02009912/n02009912_9959.JPEG', 'n02009912/n02009912_9977.JPEG',
            'n02009912/n02009912_9991.JPEG'], dtype='<U31'), 'synsets': array(['n01440764', 'n01440764', 'n01440764', ..., 'n02009912',
            'n02009912', 'n02009912'], dtype='<U9'), 'class_label': array([  0,   0,   0, ..., 132, 132, 132]), 'human_label': array(['tench, Tinca tinca', 'tench, Tinca tinca', 'tench, Tinca tinca',
            ..., 'American egret, great white heron, Egretta albus',
            'American egret, great white heron, Egretta albus',
            'American egret, great white heron, Egretta albus'], dtype='<U87')}
        >>> labels["relpath"]
        array(['n01440764/n01440764_10026.JPEG', 'n01440764/n01440764_10027.JPEG',
            'n01440764/n01440764_10029.JPEG', ...,
            'n02009912/n02009912_9959.JPEG', 'n02009912/n02009912_9977.JPEG',
            'n02009912/n02009912_9991.JPEG'], dtype='<U31')
        >>> labels["synsets"]
        array(['n01440764', 'n01440764', 'n01440764', ..., 'n02009912',
            'n02009912', 'n02009912'], dtype='<U9')
        >>> labels["class_label"]
        array([  0,   0,   0, ..., 132, 132, 132])
        >>> labels["human_label"]
        array(['tench, Tinca tinca', 'tench, Tinca tinca', 'tench, Tinca tinca',
            ..., 'American egret, great white heron, Egretta albus',
            'American egret, great white heron, Egretta albus',
            'American egret, great white heron, Egretta albus'], dtype='<U87')
        """
        self.abspaths= [s['image_path'] for s in data]
        
        classfile = os.path.dirname(os.path.realpath(self.txt_file) ) +'/classList.json'
        with open(classfile) as f:
            classList = json.load(f)

        labels = {
            "relpath": np.array([ os.path.basename(s) for s in self.abspaths]),
            "synsets": np.array([os.path.dirname(os.path.realpath(s)) for s in self.abspaths]), 
            "class_label": np.array([classList[ os.path.basename(s['image_path']) ][0] if os.path.basename(s['image_path']) in classList else 91  for s in data]),
            "human_label": np.array([" ".join(s['captions']) for s in data]),
        }

        """ print("relpath",type(labels["relpath"]))
        print("synsets",type(labels["synsets"]))
        print("class_label",type(labels["class_label"]))
        print("human_label",type(labels["human_label"])) """

        if self.process_images:
            self.size = retrieve(self.config, "size", default=128)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths
        

        
    def __getitem__(self, i):
        return self.data[i]
 
    
    def __len__(self):
        return len(self.data)


class SRBase(Dataset):
    def __init__(self,
                 txt_file,
                 state,
                 degradation=None,
                 size=None,
                 downscale_f=4,
                 min_crop_f=0.5,
                 max_crop_f=1.,
                 random_crop=True,
                 interpolation="bicubic",
                 flip_p=0.5,
                 config=None
                 ):

        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get(
            "keep_orig_class_label", False)

        self.data_root = ''
        print('began')

        with open(txt_file, 'r') as fp:
            data = json.load(fp)
        """ if state == 'val':
            self.image_paths = data[:1000]
        else:
            self.image_paths = data[1000:] """
        self.image_paths = data
        self.size = size or 256
        self.LR_size = int(self.size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)
        self.center_crop = not random_crop

        """ self.labels = {
            "relative_file_path_": [0 for l in self.image_paths],
            "file_path_": self.image_paths,
        } """
        self._length = len(self.image_paths)
        print(f'state: {state}, dataset size:{self._length}')
        self.hr_height, self.hr_width = (256, 256)

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False

        if degradation == "bsrgan":
            self.degradation_process = partial(
                degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __getitem__(self, i):
        example = {}
        image_path = self.data_root + self.image_paths[i]["image_path"]

        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * \
            np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(
                height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(
                height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        """
        image = cv2.imread(image_path.replace('teeth_info.txt','im.jpg'))
        with open(image_path,'r') as f:
            txt = f.read()
        text = txt.replace('\n','\n ')
        image = cv2.resize(image,(256,256))
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["caption"] = text
        """

        example["captions"] = " ".join(self.image_paths[i]["captions"])

        return example

    def __len__(self):
        return self.size


class TxtSRtrain(SRBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/prabhatkr/clip-sanskrit-data/annotations/jsonlinessanskrit-captions_train2017.json", state='train', **kwargs)


class TxtSRval(SRBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/prabhatkr/clip-sanskrit-data/annotations/jsonlinessanskrit-captions_val2017.json", state='val', **kwargs)


class Txttrain(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/prabhatkr/clip-sanskrit-data/annotations/jsonlinessanskrit-captions_train2017.json", state='train', **kwargs)


class Txtval(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/prabhatkr/clip-sanskrit-data/annotations/jsonlinessanskrit-captions_val2017.json", state='val', **kwargs)
