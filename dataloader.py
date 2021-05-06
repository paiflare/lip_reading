import os
import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import nltk
from torchvideotransforms import video_transforms
from itertools import islice
from joblib import Parallel, delayed


#data_path = os.path.normpath('E:/lip_reading_data/LRS3/lrs3_trainval/trainval')
#data_path_zip = os.path.normpath('E:/lip_reading_data/LRS3/lrs3_trainval/trainval.zip')

def get_vframes(video_path, transform=None, in_zip=None):
    if in_zip and in_zip.endswith('.zip'):
        video_file = zipfile.ZipFile(in_zip).open(video_path, mode='r') # считывает в виде байтов
    else:
        video_file = open(video_path, mode='rb')
    vframes, aframes, info = torchvision.io.read_video(video_file) # считывание видео [time, height, width, color]
    video_file.close()
    
    # change tensor to list of np.array
    vframes = list(map(lambda tensor: np.asarray(tensor, dtype=np.float32)/255, vframes))

    # преобразования видео (стандартные; они ожидают list of np.array)
    if transform:
        vframes = transform(vframes)

    # преобразование видео для отправки в нейросеть LipReadingNN
    # так как в LipReadinNN встроен ResNet18, то он ожидает нормализованную картинку определенного размера
    resize = video_transforms.Resize((224, 224))
    vframes = resize(vframes) # list of np.array [height=224, width=224, color]
    vframes = torch.tensor(vframes) # [time, height, width, color]
    vframes = torch.moveaxis(vframes, (0,1,2,3), (1,2,3,0)) # [color, time, height, width]
    normalize = video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    vframes = normalize(vframes) # [color, time, height, width]
    vframes = torch.moveaxis(vframes, (0,1), (1,0)) # [time, color, height, width]
    
    return vframes

def get_tokens(txt_path, tokenizer=None, in_zip=None):
    if in_zip and in_zip.endswith('.zip'):
        txt_file = zipfile.ZipFile(in_zip).open(txt_path, mode='r') # считывает в виде байтов
        _, text = txt_file.readline().decode('utf-8').split(':') # прочитанный текст
    else:
        txt_file = open(txt_path, mode='r')
        _, text = txt_file.readline().split(':') # прочитанный текст
    text = text.strip()
    txt_file.close()
    
    if tokenizer:
        tokens = tokenizer.tokenize(text)
    else:
        tokens = text.split()
    
    return tokens

class LipReadingVideoDataset(Dataset):
    def __init__(self, data_path, transform=None, tokenizer=None):
        #self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        
        self.file_paths = []
        if data_path.endswith('.zip'): # если путь указан до .zip
            file_zip = zipfile.ZipFile(data_path)
            # перебрать файлы
            for file in file_zip.filelist:
                file_path = file.filename
                if file_path.endswith('.mp4'):
                    self.file_paths.append(file_path)
        else: # если путь указан до директории
            for dir_path, dir_names, file_names in os.walk(data_path):
                # перебрать файлы
                for file_name in file_names:
                    if file_name.endswith('.mp4'):
                        file_path = os.path.join(dir_path, file_name)
                        self.file_paths.append(file_path)
        self.data_path = data_path # путь либо до директории, либо до zip файла
        
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer is not None else nltk.tokenize.WordPunctTokenizer()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        vframes = get_vframes(video_path, transform=self.transform, in_zip=self.data_path)

        txt_path = video_path.replace('.mp4', '.txt')  # заменяем путь на путь до видео
        tokens = get_tokens(txt_path, tokenizer=self.tokenizer, in_zip=self.data_path)

        sample = {'vframes': vframes, 'text': tokens}

        return sample

# функция создания батча
def collate_func(list_of_samples):
    cut = 128
    max_time = min(cut, max(sample['vframes'].shape[0] for sample in list_of_samples))
    win_len = 16
    hop_len = 8
    n_pad = int(win_len // 2)
    
    def create_temp_vframes(sample):
        vframes, tokens = sample.values()
        vframes = vframes[:cut]
        #list_of_tokens.append(tokens)
        # vframes has shape (time, color, height, width)

        # padding vframes with zeros along 'time' axis untill all samples has same max_time in batch
        # after this vframes has shape (max_time, color, height, width)
        time = vframes.shape[0]
        temp_vframes = np.pad(vframes, ((0,max_time-time), (0,0), (0,0), (0,0)))

        # paddin vframes with ... along 'time' axis
        # after this vframes has shape (max_time + 2*n_pad, color, height, width)
        temp_vframes = np.pad(temp_vframes, ((n_pad,n_pad), (0,0), (0,0), (0,0)))

        # now crop vframes along 'time' axis with overlap
        # after this vframes has shape (frames_num, win_len(this is time axis), color, height, width)
        max_time_long = max_time + 2*n_pad
        temp_vframes = np.stack(np.array(tuple(islice(temp_vframes, i, i+win_len))) for i in range(0, max_time_long-win_len, hop_len))

        return temp_vframes, tokens
    
    res = Parallel(n_jobs=8)(delayed(create_temp_vframes)(sample) for sample in list_of_samples)
    
    new_vframes = []
    list_of_tokens = []
    for temp_vframes, tokens in res:
        new_vframes.append(temp_vframes)
        list_of_tokens.append(tokens)

    vframes_batch = torch.tensor(new_vframes, dtype=torch.float32) # shape (batch_size, frames_num, win_len, color, height, width)

    return vframes_batch, list_of_tokens