from scipy.io import savemat, loadmat
import time
import numpy as np
import trimesh

"""
modified from https://github.com/soubhiksanyal/RingNet/blob/master/dynamic_contour_embedding.py
"""

import os, argparse
import numpy as np
import trimesh
import _pickle as pickle
from Get_landmarks import get_landmarks


def generate_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    points_neutral = []
    points_target = []
    landmarks_target = []
    landmarks_neutral = []

    vertices = np.load(args.vertices_path, allow_pickle=True)
    templates = np.load(args.templates_path, allow_pickle=True)

    for i in range(len(vertices)):
        print(i)
        points_neutral.append(templates[i])
        points_target.append(vertices[i])
        landmarks_neutral.append(get_landmarks(templates[i]))
        landmarks_target.append(get_landmarks(vertices[i]))


    print(np.shape(points_neutral))
    print(np.shape(points_target))

    if not os.path.exists(os.path.join(args.save_path, 'points_input')):
        os.makedirs(os.path.join(args.save_path, 'points_input'))

    if not os.path.exists(os.path.join(args.save_path, 'points_target')):
        os.makedirs(os.path.join(args.save_path, 'points_target'))

    if not os.path.exists(os.path.join(args.save_path, 'landmarks_target')):
        os.makedirs(os.path.join(args.save_path, 'landmarks_target'))

    if not os.path.exists(os.path.join(args.save_path, 'landmarks_input')):
        os.makedirs(os.path.join(args.save_path, 'landmarks_input'))

    for j in range(len(points_neutral)):
        np.save(os.path.join(args.save_path, 'points_input', '{0:08}_frame'.format(j)), points_neutral[j])
        np.save(os.path.join(args.save_path, 'points_target', '{0:08}_frame'.format(j)), points_target[j])
        np.save(os.path.join(args.save_path, 'landmarks_target', '{0:08}_frame'.format(j)), landmarks_target[j])
        np.save(os.path.join(args.save_path, 'landmarks_input', '{0:08}_frame'.format(j)), landmarks_neutral[j])

    files = []
    for r, d, f in os.walk(os.path.join(args.save_path, 'points_input')):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(args.save_path, 'paths_test.npy'), files)

    files = []
    for r, d, f in os.walk(os.path.join(args.save_path, 'landmarks_target')):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(args.save_path, 'landmarks_test.npy'), files)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices_path", type=str, default="./coma_florence/coma_vertices.npy")
    parser.add_argument("--templates_path", type=str, default="./coma_florence/coma_templates.npy")
    parser.add_argument("--save_path", type=str, default='./coma_florence/training_data')


    args = parser.parse_args()

    generate_dataset(args)

if __name__ == '__main__':
    main()



