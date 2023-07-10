import numpy as np
import argparse
from tqdm import tqdm
import os, shutil

import torch
import torch.nn as nn

from data_loader import get_dataloaders
from model import Speech2Land


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    iteration = 0
    for e in range(epoch + 1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, landmarks, template, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, landmarks, template = audio.to(device=args.device), landmarks.to(device=args.device), template.to(device=args.device)
            loss = model(audio, landmarks - template, criterion)
            loss.backward()
            loss_log.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))
        # validation
        valid_loss_log = []
        model.eval()
        for audio, landmarks,  template, file_name in dev_loader:
            # to gpu
            audio, landmarks, template = audio.to(device=args.device), landmarks.to(device=args.device), template.to(device=args.device)

            loss = model(audio, landmarks - template, criterion)
            valid_loss_log.append(loss.item())


        current_loss = np.mean(valid_loss_log)

        if e == args.max_epoch or (e + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_S2L.pth'.format(e + 1)))

        print("epoch: {}, current loss:{:.8f}".format(e + 1, current_loss))
    return model


@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    save_path = args.save_path

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_S2L.pth'.format(epoch))))
    model = model.to(torch.device(args.device))
    model.eval()

    for audio, landmarks, template, file_name in test_loader:
        # to gpu
        audio, landmarks, template = audio.to(device=args.device), landmarks.to(device=args.device), template.to(device=args.device)

        prediction = model.predict(audio, template)
        prediction = prediction.squeeze()  # (seq_len, V*3)
        np.save(os.path.join(result_path, file_name[0].split(".")[0] + ".npy"),
                prediction.detach().cpu().numpy())



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Weighted_Loss(nn.Module):
    def __init__(self, args):
        super(Weighted_Loss, self).__init__()
        self.weights = torch.zeros(204)
        self.weights[15:39] = 1
        self.weights[144:] = 1
        self.weights = self.weights.to(args.device)
        self.mse = nn.MSELoss(reduction='none')
        self.cos_sim = nn.CosineSimilarity(dim=3)

    def forward(self, predictions, target):
        rec_loss = torch.mean(self.mse(predictions, target))

        mouth_rec_loss = torch.sum(self.mse(predictions * self.weights, target * self.weights)) / (
                    84 * predictions.shape[1])

        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean(
            (self.mse(prediction_shift, target_shift)))

        len = predictions.shape[1]
        predictions = torch.reshape(predictions, (1, len, 68, 3))
        target = torch.reshape(target, (1, len, 68, 3))

        cos_dist = torch.mean((1 - self.cos_sim(predictions, target)))

        return 0.1 * rec_loss + 1 * mouth_rec_loss + 10 * vel_loss + 0.0001 * cos_dist

def main():
    parser = argparse.ArgumentParser(description='Speech2Land: Speech-Driven 3D Landmarks generation with S2L')
    parser.add_argument("--lr", type=float, default=0.00005, help='learning rate')
    parser.add_argument("--landmarks_dim", type=int, default=68*3, help='number of landmarks - 68*3')
    parser.add_argument("--audio_feature_dim", type=int, default=768, help='768 for wav2vec')
    parser.add_argument("--feature_dim", type=int, default=64, help='64')
    parser.add_argument("--wav_path", type=str, default="vocaset/wav", help='path of the audio signals')
    parser.add_argument("--landmarks_path", type=str, default="vocaset/landmarks_npy", help='path of the ground truth')
    parser.add_argument("--max_epoch", type=int, default=300, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="vocaset/templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="Saves", help='')
    parser.add_argument("--result_path", type=str, default="Results", help='path to the predictions')
    parser.add_argument("--num_layers", type=int, default=3, help='number of S2L layers')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    # build model
    model = Speech2Land(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device(args.device))

    # loss
    criterion = Weighted_Loss(args)

    # load data
    dataset = get_dataloaders(args)

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)

    test(args, model, dataset["test"], epoch=args.max_epoch)


if __name__ == "__main__":
    main()