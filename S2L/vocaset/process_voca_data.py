import os
import argparse
import pickle
import numpy as np
from scipy.io import wavfile
from Get_landmarks import get_landmarks

def load_data(args):
    face_vert_mmap = np.load(args.vertices_path, mmap_mode='r+')
    raw_audio = pickle.load(open(args.raw_audio_path, 'rb'), encoding='latin1')
    data2array_verts = pickle.load(open(args.data2array_verts_path, 'rb'))
    return face_vert_mmap,raw_audio,data2array_verts

def generate_landmarks_npy(args,face_vert_mmap,data2array_verts):
    if not os.path.exists(args.landmarks_npy_path):
        os.makedirs(args.landmarks_npy_path)
    for sub in data2array_verts.keys():
        for seq in data2array_verts[sub].keys():
            landmarks_npy_name = sub + "_" + seq
            landmarks_npy = []
            for frame, array_idx in data2array_verts[sub][seq].items():
                landmarks_npy.append(get_landmarks(face_vert_mmap[array_idx]))
            landmarks_npy = np.array(landmarks_npy).reshape(-1,args.landmarks_dim)
            np.save(os.path.join(args.landmarks_npy_path, landmarks_npy_name), landmarks_npy)

def generate_vertices_npy(args,face_vert_mmap,data2array_verts):
    if not os.path.exists(args.vertices_npy_path):
        os.makedirs(args.vertices_npy_path)
    for sub in data2array_verts.keys():
        for seq in data2array_verts[sub].keys():
            vertices_npy_name = sub + "_" + seq
            vertices_npy = []
            for frame, array_idx in data2array_verts[sub][seq].items():
                vertices_npy.append(face_vert_mmap[array_idx])
            vertices_npy = np.array(vertices_npy).reshape(-1,args.vertices_dim)
            np.save(os.path.join(args.vertices_npy_path, vertices_npy_name), vertices_npy)

def generate_wav(args,raw_audio):
    if not os.path.exists(args.wav_path):
        os.makedirs(args.wav_path)
    for sub in raw_audio.keys():
        for seq in raw_audio[sub].keys():
            wav_name = sub + "_" + seq
            wavfile.write(os.path.join(args.wav_path, wav_name+'.wav'), raw_audio[sub][seq]['sample_rate'], raw_audio[sub][seq]['audio'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices_path", type=str, default="data_verts.npy")
    parser.add_argument("--landmarks_npy_path", type=str, default="landmarks_npy")
    parser.add_argument("--landmarks_dim", type=int, default=68*3)
    parser.add_argument("--raw_audio_path", type=str, default='raw_audio_fixed.pkl')
    parser.add_argument("--wav_path", type=str, default='wav')
    parser.add_argument("--data2array_verts_path", type=str, default='subj_seq_to_idx.pkl')
    args = parser.parse_args()

    face_vert_mmap, raw_audio, data2array_verts = load_data(args)
    generate_landmarks_npy(args, face_vert_mmap, data2array_verts)
    generate_wav(args, raw_audio)

if __name__ == '__main__':
    main()