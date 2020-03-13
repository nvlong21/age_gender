import os
import cv2
import re
import glob
import math
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class FaceDataset(Dataset):
  """ read images from disk dynamically """

  def __init__(self,transformer, pharse = "train"):
    """
    init function
    :param datapath: datapath to aligned folder  
    :param transformer: image transformer
    """
    self.pics         = glob.glob("/media/hdd/sources/data/tmp/%s/"%pharse + "*.jpg")
    self.transformer  = transformer
    self.age_divde = float(10)
    self.age_cls_unit = int(101)

    self.age_cls = {x: self.GaussianProb(x) for x in range(1, self.age_cls_unit + 1)}
    self.age_cls_zeroone = {x: self.ZeroOneProb(x) for x in range(1, self.age_cls_unit + 1)}

  def __len__(self):
    return len(self.pics)

  def GaussianProb(self, true, var = 2.5):
    x = np.array(range(1, self.age_cls_unit + 1), dtype='float')
    probs = np.exp(-np.square(x - true) / (2 * var ** 2)) / (var * (2 * np.pi ** .5))
    return probs / probs.max()

  def ZeroOneProb(self, true):
    x = np.zeros(shape=(self.age_cls_unit, ))
    x[true - 1] = 1
    return x


  def __getitem__(self, idx):
    """
    get images and labels
    :param idx: image index 
    :return: image: transformed image, gender: torch.LongTensor, age: torch.FloatTensor
    """
    # read image and labels
    img_name = self.pics[idx]
    img = cv2.imread(img_name)
    if len(img.shape) == 2: # gray image
      img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    try:
        _, age, gender = self.pics[idx].replace(".pg", "").split("/")[-1].split('.')[0].split('_')
    except:
        print(self.pics[idx])
    age = max(1., min(float(age), float(self.age_cls_unit)))

    # preprcess images
    if self.transformer:
      img = transforms.ToPILImage()(img)
      image = self.transformer(img)
    else:
      image = torch.from_numpy(img)

    # preprocess labels
    gender = float(gender)
    gender = torch.from_numpy(np.array([gender], dtype='float'))
    gender = gender.type(torch.LongTensor)

    age_rgs_label = torch.from_numpy(np.array([age / self.age_divde], dtype='float'))
    age_rgs_label = age_rgs_label.type(torch.FloatTensor)

    age_cls_label = self.age_cls[int(age)]
    # age_cls_label = self.age_cls_zeroone[int(age)]

    age_cls_label = torch.from_numpy(np.array([age_cls_label], dtype='float'))
    age_cls_label = age_cls_label.type(torch.FloatTensor)

    # image of shape [256, 256]
    # gender of shape [,1] and value in {0, 1}
    # age of shape [,1] and value in [0 ~ 10)
    return image, gender, age_rgs_label, age_cls_label



