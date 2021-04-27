import argparse
import os
import torch
import pickle

from network_small import LipReadinNN_LSTM
from dataloader import get_vframes, collate_func

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to .mp4 file for lip reading", type=str)
    parser.add_argument("--gpu", help="switch to gpu", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    video_path = os.path.normpath(args.path)

    # считываем файл
    vframes = get_vframes(video_path)
    dummy_sample = {'vframes': vframes, 'text': []}
    dummy_batch = collate_func([dummy_sample])

    # загружаем словарь
    # десериализация словаря
    with open('token_to_id.pickle', 'rb') as file:
        token_to_id = pickle.load(file)

    # загружаем модель
    PATH = 'models/model_.pt'
    device = torch.device('cpu' if not (torch.cuda.is_available() and args.gpu) else 'cuda')

    lip_reading_nn = LipReadinNN_LSTM(token_to_id)
    lip_reading_nn.load_state_dict(torch.load(PATH))
    lip_reading_nn.to(device)

    output = lip_reading_nn(dummy_batch)

    print(output)