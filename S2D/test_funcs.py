import torch
import copy
from tqdm import tqdm
import numpy as np


def test_autoencoder_dataloader(device, model, dataloader_test, shapedata):
    model.eval()
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            lands = sample_dict['landmarks'].to(device)
            target = sample_dict['points_target'].to(device)
            prediction, displacement = model(tx, lands)
            if i == 0:
                predictions = copy.deepcopy(prediction)
                displacements = copy.deepcopy(displacement)
                inputs = copy.deepcopy(tx)
                landmarks = copy.deepcopy(lands)
                targets = copy.deepcopy(target)
            else:
                predictions = torch.cat([predictions, prediction], 0)
                displacements = torch.cat([displacements, displacement], 0)
                landmarks = torch.cat([landmarks, lands], 0)
                inputs = torch.cat([inputs, tx], 0)
                targets = torch.cat([targets, target], 0)

        predictions = predictions.cpu().numpy()
        displacements = displacements.cpu().numpy()
        landmarks = landmarks.cpu().numpy()
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        if shapedata.normalization:
            for jj in range(np.shape(predictions)[0]):
                predictions[jj, :-1, :] = predictions[jj, :-1, :] * shapedata.std + shapedata.mean

    return predictions, inputs, landmarks, targets