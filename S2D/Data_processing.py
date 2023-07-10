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

def load_data(args):
    face_vert_mmap = np.load(args.vertices_path, mmap_mode='r+')
    data2array_verts = pickle.load(open(args.data2array_verts_path, 'rb'))

    with open(args.template_path, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    return face_vert_mmap,data2array_verts, templates



def generate_dataset(args,face_vert_mmap,data2array_verts, templates):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    points_neutral = []
    points_target = []
    landmarks_target = []
    landmarks_neutral = []

    for sub in data2array_verts.keys():
        for seq in data2array_verts[sub].keys():
            for frame, array_idx in data2array_verts[sub][seq].items():
                points_neutral.append(templates[sub])
                points_target.append(np.array(face_vert_mmap[array_idx]))
                landmarks_neutral.append(get_landmarks(templates[sub]))
                landmarks_target.append(get_landmarks(face_vert_mmap[array_idx]))

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
    parser.add_argument("--vertices_path", type=str, default="./vocaset/data_verts.npy")
    parser.add_argument("--data2array_verts_path", type=str, default='./vocaset/subj_seq_to_idx.pkl')
    parser.add_argument("--template_path", type=str, default='./vocaset/templates.pkl')
    parser.add_argument("--save_path", type=str, default='./vocaset/training_data')

    args = parser.parse_args()


    face_vert_mmap, data2array_verts, templates = load_data(args)
    generate_dataset(args, face_vert_mmap, data2array_verts, templates)

if __name__ == '__main__':
    main()



