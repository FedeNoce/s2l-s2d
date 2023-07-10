from torch.utils.data import Dataset
import torch
import numpy as np
import os

class autoencoder_dataset(Dataset):

    def __init__(self, template, neutral_root_dir, points_dataset, shapedata, normalization=True, dummy_node=True):
        # points_dataset: train/val/test

        self.shapedata = shapedata
        self.normalization = normalization
        self.neutral_root_dir = neutral_root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(neutral_root_dir, 'paths_' + points_dataset + '.npy'))
        self.template = template
        self.paths_lands = np.load(os.path.join(neutral_root_dir, 'landmarks_test.npy'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        basename_landmarks = self.paths_lands[idx]

        verts_input = np.load(os.path.join(self.neutral_root_dir, 'points_input', basename + '.npy'), allow_pickle=True)
        if os.path.isfile(os.path.join(self.neutral_root_dir, 'points_target', basename + '.npy')):
            verts_target = np.load(os.path.join(self.neutral_root_dir, 'points_target', basename + '.npy'),
                                   allow_pickle=True)
        else:
            verts_target = np.zeros(np.shape(verts_input))

        landmarks_neutral = np.load(os.path.join(self.neutral_root_dir, 'landmarks_input', basename_landmarks + '.npy'),
                                    allow_pickle=True)
        landmarks = np.load(os.path.join(self.neutral_root_dir, 'landmarks_target', basename_landmarks + '.npy'),
                            allow_pickle=True)
        landmarks = landmarks - landmarks_neutral

        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init / self.shapedata.std
            verts_neutral = verts_neutral - self.shapedata.mean
            verts_neutral = verts_neutral / self.shapedata.std

        verts_input[np.where(np.isnan(verts_input))] = 0.0

        verts_input = verts_input.astype('float32')

        landmarks = landmarks.astype('float32')

        if self.dummy_node:
            verts_ = np.zeros((verts_input.shape[0] + 1, verts_input.shape[1]), dtype=np.float32)
            verts_[:-1, :] = verts_input

            verts_input = verts_

        verts_input = torch.Tensor(verts_input)
        landmarks = torch.Tensor(landmarks)

        sample = {'points': verts_input, 'landmarks': landmarks, 'points_target': verts_target}

        return sample
