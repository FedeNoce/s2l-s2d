import os
import models
import spiral_utils
import shape_data
import autoencoder_dataset
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from train_funcs import train_autoencoder_dataloader
import argparse


def Weighted_Loss(outputs, targets, inputs, displacement, device):
    outputs = outputs.float()
    mse = nn.MSELoss(reduction='none')
    cos_sim = nn.CosineSimilarity(dim=2)

    weights = np.load('./template/template/Normalized_d_weights.npy', allow_pickle=True)
    Weigths = torch.from_numpy(weights[:-1]).float().to(device)
    inputs = inputs[:, :-1, :]
    target_expression = outputs - inputs
    displacement = displacement[:, :-1, :]
    targets = targets[:, :-1, :]

    cos_similarity = cos_sim(displacement, target_expression)
    cos_dist = torch.mean((1 - cos_similarity))

    L = (torch.matmul(Weigths, mse(outputs, targets))).mean() + 0.1 * mse(
        target_expression, displacement).mean() + 0.0001 * cos_dist

    return L


def train(args):
    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    nz = 16
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    nbr_landmarks = 68
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]
    device = args.device
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                          test_file=args.root_dir + '/test.npy',
                          reference_mesh_file=args.reference_mesh_file,
                          normalization=False,
                          meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3


    with open('./template/template/downsampling_matrices.pkl', 'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in range(len(M_verts_faces))]

    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    for i in range(len(ds_factors)):
        dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    Adj, Trigs = spiral_utils.get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage='trimesh')

    spirals_np, spiral_sizes, spirals = spiral_utils.generate_spirals(step_sizes,
                                                            M, Adj, Trigs,
                                                            reference_points = reference_points,
                                                            dilation = dilation, random = False,
                                                            meshpackage = 'trimesh',
                                                            counter_clockwise = True)
    sizes = [x.vertices.shape[0] for x in M]


    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)


    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]



    dataset_train = autoencoder_dataset.autoencoder_dataset(neutral_root_dir=args.root_dir, points_dataset='test',
                                       shapedata=shapedata,
                                       normalization=False, template=args.reference_mesh_file)

    dataloader_train = DataLoader(dataset_train, batch_size=16,
                                 shuffle=False, num_workers=4)


    my_decoder = models.SpiralAutoencoder(filters_enc=filter_sizes_enc,
                                      filters_dec=filter_sizes_dec,
                                      latent_size=nz,
                                      sizes=sizes,
                                      nbr_landmarks=nbr_landmarks,
                                      spiral_sizes=spiral_sizes,
                                      spirals=tspirals,
                                      D=tD, U=tU, device=device).to(device)

    loss_fn = Weighted_Loss

    optim = torch.optim.Adam(my_decoder.parameters(), lr=args.lr)

    train_autoencoder_dataloader(dataloader_train, device, my_decoder, optim, loss_fn, 0, args.epochs, args.result_dir)

def main():
    parser = argparse.ArgumentParser(description='S2D: Sparse to Dense Decoder')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--reference_mesh_file", type=str, default='./template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--epochs", type=int, default=1000, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--root_dir", type=str, default='./vocaset/training_data')
    parser.add_argument("--result_dir", type=str, default='./Results')

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
