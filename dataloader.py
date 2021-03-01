from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import cv2 as cv

data_path = os.path.normpath('E:/lip_reading_data/LRS3/lrs3_trainval/trainval')
answers_path = os.path.normpath('E:/lip_reading_data/LRS3/lrs3_v0.4_txt/lrs3_v0.4/trainval')

class LipReadingVideoDataset(Dataset):

    def __init__(self, answers_path, data_path, transform=None):
        self.file_paths = []
        for dir_path, dir_names, file_names in os.walk(answers_path):
            # перебрать файлы
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                self.file_paths.append(file_path)
        self.answers_path = answers_path
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        txt_path = self.file_paths[idx]
        txt_file = open(txt_path, 'r')
        _, text = txt_file.readline().split(':')
        text = text.strip()

        _, conf = txt_file.readline().split(':')
        conf = conf.strip()

        _, ref = txt_file.readline().split(':')
        ref = ref.strip()

        txt_file.readline()

        face_df = pd.read_csv(txt_file, sep='\t')
        txt_file.close()

        head, tail = os.path.split(txt_path)
        head = head.replace(answers_path, data_path)
        tail = tail.replace('txt', 'mp4').replace('0', '5', 1)
        video_path = os.path.join(head, tail)

        vframes, aframes, info = torchvision.io.read_video(video_path)

        sample = {'vframes': vframes, 'text': text, 'face': face_df}

        if self.transform:
            sample = self.transform(sample)

        return sample