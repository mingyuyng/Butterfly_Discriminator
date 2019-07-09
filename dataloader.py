import torch
import torch.utils.data as data
import scipy.io as sio
import os
import glob


class dataLoader(data.Dataset):

    def __init__(self, root):

        self.root = root
        self.num = len(glob.glob1(self.root, "*.mat"))

    def __getitem__(self, index):

        lis = [self.root, str(index) + '.mat']
        filename = os.path.join(*lis)

        mat_data = sio.loadmat(filename)

        intensity = mat_data['intensity'].astype('double')
        label = mat_data['label'][0, 0]

        pair = {'intensity': intensity, 'label': label}
        return pair

    def __len__(self):
        return self.num
