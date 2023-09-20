import S2D.models as models
import S2D.spiral_utils as spiral_utils
import S2D.shape_data as shape_data
import S2D.autoencoder_dataset as autoencoder_dataset
import S2D.save_meshes as save_meshes
import argparse
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from S2D.test_funcs import test_autoencoder_dataloader
import torch
import PySimpleGUI as sg
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import cv2
import Get_landmarks
from S2L.model import Speech2Land
from transformers import Wav2Vec2Processor
import PIL
import time
import os
import cv2
import tempfile
import numpy as np
from subprocess import call
from psbody.mesh import Mesh
import pyrender
import trimesh
import glob
import librosa
from torch.utils.data import DataLoader
import vlc
from sys import platform as PLATFORM
from scipy.io import wavfile
import scipy.misc
from Rec.FacePartsSwap.face_swap import rec


os.environ['PYOPENGL_PLATFORM'] = 'egl'
def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')


def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):

    background_black = True
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    if background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])  # [0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2],
                               bg_color=[255, 255, 255])  # [0, 0, 0] black,[255, 255, 255] white

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(window, audio_fname, sequence_vertices, template, out_path , out_fname, fps, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for i_frame in range(num_frames - 2):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
        cv2.imwrite('/home/federico/Scrivania/TH/Demos/template.png', img)
        cv2.imwrite('/home/federico/Scrivania/TH/Images/template' + str(i) + '.png', img)
        window["-IMAGE_PREVIEW-"].update(filename='/home/federico/Scrivania/TH/Demos/template.png')
        window.refresh()
        i = i + 1
    writer.release()

    video_fname = os.path.join(out_path, out_fname)
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)

def generate_mesh_video(out_path, out_fname, meshes_path_fname, fps, audio_fname, template, window):

    sequence_fnames = sorted(glob.glob(os.path.join(meshes_path_fname, '*.ply*')))

    audio_fname = audio_fname


    uv_template_fname = template
    sequence_vertices = []
    f = None

    for frame_idx, mesh_fname in enumerate(sequence_fnames):
        frame = Mesh(filename=mesh_fname)
        sequence_vertices.append(frame.v)
        if f is None:
            f = frame.f

    template = Mesh(sequence_vertices[0], f)
    sequence_vertices = np.stack(sequence_vertices)
    render_sequence_meshes(window, audio_fname, sequence_vertices, template, out_path, out_fname, fps, uv_template_fname=uv_template_fname, texture_img_fname='')


def generate_landmarks(args, model_path, audio_path, template_file, save_path, template_name):

    model = Speech2Land(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model = model.to(torch.device(args.device))
    model.eval()

    speech_array, sampling_rate = librosa.load(os.path.join(audio_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
    print(template_name)
    if template_name != 'Reconstruction':
        with open(template_file, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')
        actor_vertices = templates[template_name]
    else:
        actor = trimesh.load('/home/federico/Scrivania/TH/Rec/reconstruction.ply')
        actor_vertices = actor.vertices
        
    actor_landmarks = Get_landmarks.get_landmarks(actor_vertices)
    actor = actor_landmarks.reshape((-1))
    actor = np.reshape(actor, (-1, actor.shape[0]))
    actor = torch.FloatTensor(actor).to(device=args.device)

    prediction = model.predict(audio_feature, actor)
    prediction = prediction.squeeze()  # (seq_len, V*3)
    landmarks = prediction.detach().cpu().numpy()
    landmarks = np.reshape(landmarks, (landmarks.shape[0], 68, 3))

    if not os.path.exists(os.path.join(save_path, 'points_input')):
        os.makedirs(os.path.join(save_path, 'points_input'))

    if not os.path.exists(os.path.join(save_path, 'points_target')):
        os.makedirs(os.path.join(save_path, 'points_target'))

    if not os.path.exists(os.path.join(save_path, 'landmarks_target')):
        os.makedirs(os.path.join(save_path, 'landmarks_target'))

    if not os.path.exists(os.path.join(save_path, 'landmarks_input')):
        os.makedirs(os.path.join(save_path, 'landmarks_input'))

    for j in range(len(landmarks)):
                np.save(os.path.join(save_path, 'points_input', '{0:08}_frame'.format(j)), actor_vertices)
                np.save(os.path.join(save_path, 'points_target', '{0:08}_frame'.format(j)), actor_vertices)
                np.save(os.path.join(save_path, 'landmarks_target', '{0:08}_frame'.format(j)), landmarks[j])
                np.save(os.path.join(save_path, 'landmarks_input', '{0:08}_frame'.format(j)), actor_landmarks)

    files = []

    for r, d, f in os.walk(os.path.join(save_path, 'points_input')):
                for file in f:
                    if '.npy' in file:
                        files.append(os.path.splitext(file)[0])
    np.save(os.path.join(save_path, 'paths_test.npy'), sorted(files))

    files = []
    for r, d, f in os.walk(os.path.join(save_path, 'landmarks_target')):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(save_path, 'landmarks_test.npy'), sorted(files))
    print('Done')

def generate_meshes_from_landmarks(template_path, reference_mesh_path, landmarks_path, prediction_path, save_meshes_path, args):


    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    nz = 16
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    nbr_landmarks = 68
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                                     test_file=landmarks_path + '/test.npy',
                                     reference_mesh_file=reference_mesh_path,
                                     normalization=False,
                                     meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    with open(
            'S2D/template/template/downsampling_matrices.pkl',
            'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
         range(len(M_verts_faces))]

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
                                                                      reference_points=reference_points,
                                                                      dilation=dilation, random=False,
                                                                      meshpackage='trimesh',
                                                                      counter_clockwise=True)

    sizes = [x.vertices.shape[0] for x in M]

    device = torch.device(args.device)

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
    dataset_test = autoencoder_dataset.autoencoder_dataset(neutral_root_dir=landmarks_path, points_dataset='test',
                                                           shapedata=shapedata,
                                                           normalization=False, template=reference_mesh_path)

    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=False, num_workers=4)

    model = models.SpiralAutoencoder(filters_enc=filter_sizes_enc,
                                     filters_dec=filter_sizes_dec,
                                     latent_size=nz,
                                     sizes=sizes,
                                     nbr_landmarks=nbr_landmarks,
                                     spiral_sizes=spiral_sizes,
                                     spirals=tspirals,
                                     D=tD, U=tU, device=device).to(device)


    checkpoint = torch.load(args.S2D, map_location=device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    predictions, inputs, lands, targets = test_autoencoder_dataloader(device, model, dataloader_test, shapedata)
    np.save(os.path.join(prediction_path, 'targets'), targets)
    np.save(os.path.join(prediction_path, 'predictions'), predictions)
    save_meshes.save_meshes(predictions, save_meshes_path, n_meshes=len(predictions), template_path=template_path)
    print('Done')


def main():
    parser = argparse.ArgumentParser(description='S2L+S2D: Speech-Driven 3D Talking heads')
    parser.add_argument("--landmarks_dim", type=int, default=68 * 3, help='number of landmarks - 68*3')
    parser.add_argument("--audio_feature_dim", type=int, default=768, help='768 for wav2vec')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_layers", type=int, default=3, help='number of S2L layers')
    parser.add_argument("--S2L", type=str, default='S2L/Results/301_s2l.pth', help='path to the S2L model')
    parser.add_argument("--S2D", type=str, default='S2D/Results/s2d.pth.tar', help='path to the S2D model')
    parser.add_argument("--template_file", type=str, default="S2L/vocaset/templates.pkl", help='faces to animate')
    parser.add_argument("--flame_template", type=str, default="S2L/vocaset/flame_model/FLAME_sample.ply", help='template_path')
    parser.add_argument("--video_name", type=str, default="example.mp4", help='name of the rendered video')
    parser.add_argument("--fps", type=int, default=60, help='frames per second')
    
    template_names = [
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170908_03277_TA",
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170912_03278_TA",
    "FaceTalk_170731_00024_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170809_00138_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170915_00223_TA"
    ]
    audio_names = [
    "photo.wav",
    "italian.wav",
    "watch.wav",
    "military.wav",
    "sample1.wav",
    "language.wav",
    "power.wav",
    "pain.wav",
    "ted_ita.wav"
    ]
    
    #sg.theme_previewer()

    sg.set_options(font=('Arial', 14))
    sg.theme('DarkGrey15')

    
    layout = [
        [sg.Text("Demo Name:"), sg.Input(key="-DEMO_NAME-")],
        [sg.Text("Actor Name:"), sg.Combo(template_names, key="-TEMPLATE_NAME-", enable_events=True), sg.Button("Shoot a Photo"), sg.Text(" ", key="-OUTPUT_PHOTO-")],
        [sg.Text("Audio:"), sg.Combo(audio_names, key="-AUDIO_PATH-", enable_events=True), sg.Button("Record Audio"), sg.Text(" ", key="-OUTPUT_AUDIO-")],
        [sg.Button("Start Demo"), sg.Text(" ", key="-OUTPUT_DEMO-")],
        [sg.Image(key="-IMAGE_PREVIEW-", size=(800, 800)), sg.Image(key="-VIDEO_PREVIEW-", size=(800, 800))],
        [sg.Button("Play"), sg.Button("Pause"), sg.Button("Exit")]
    ]

    window = sg.Window("S2L + S2D Demo", layout).finalize()
    window.Maximize()

    recorded_audio_path = None
    recording = False

    def start_recording():
        global recorded_audio_path, recording
        recording = True
        #recorded_audio_path = tempfile.mktemp(suffix=".wav")
        recorded_audio_path = '/home/federico/Scrivania/TH/Audios/record.wav'
        fs = 44100  # Sample rate
        duration = 3  # Recording duration
        window["-OUTPUT_AUDIO-"].update('Recording started...')
        window.refresh()
        print("Recording started...")
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        wavfile.write(recorded_audio_path, fs, myrecording)

        print("Recording saved:", recorded_audio_path)
        window["-OUTPUT_AUDIO-"].update('Recording Saved!')
        window.refresh()
        window["-AUDIO_PATH-"].update(recorded_audio_path.split('/')[-1])

    def stop_recording():
        global recording
        recording = False

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        if event == "-TEMPLATE_NAME-":
            selected_template_name = values["-TEMPLATE_NAME-"]
            # Do something with the selected template name if needed
        if event == 'Shoot a Photo':
            window["-OUTPUT_PHOTO-"].update('Strike a Pose ...')
            window["-TEMPLATE_NAME-"].update('Reconstruction')
            window.refresh()
            rec()
            window["-OUTPUT_PHOTO-"].update('Face Reconstruction Completed!')
            window.refresh()
        if event == "Record Audio":
            start_recording()
        if event == "Stop Recording":
            stop_recording()

        if event == "Start Demo":
            window["-OUTPUT_DEMO-"].update('Start Demo')
            template_name = values["-TEMPLATE_NAME-"]
            audio_path = values["-AUDIO_PATH-"]
            demo_name = values["-DEMO_NAME-"]
            if recording:
                sg.popup("Please stop recording before starting the demo.")
                continue
            if not recording:
                audio_path = os.path.join('Audios', audio_path)
                #sg.popup("Invalid audio path.")
                #continue
                
            window['-VIDEO_PREVIEW-'].expand(True, True) 

            inst = vlc.Instance()
            list_player = inst.media_list_player_new()
            media_list = inst.media_list_new([])
            list_player.set_media_list(media_list)
            player = list_player.get_media_player()
            if PLATFORM.startswith('linux'):
                player.set_xwindow(window['-VIDEO_PREVIEW-'].Widget.winfo_id())
            else:
                player.set_hwnd(window['-VIDEO_PREVIEW-'].Widget.winfo_id())

            args = parser.parse_args()
            fps = args.fps
  
            save_path = 'Demos/' + demo_name

            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, 'Landmarks'))
            os.mkdir(os.path.join(save_path, 'Meshes'))
            os.mkdir(os.path.join(save_path, 'predicted_meshes'))

            model_path = args.S2L

            save_landmarks_path = os.path.join(save_path, 'Landmarks')
            actors_file = args.template_file

            template_path = args.flame_template
            prediction_path = os.path.join(save_path, 'Meshes')
            save_path_meshes = os.path.join(save_path, 'predicted_meshes')

            print('Landmarks generation')
            window["-OUTPUT_DEMO-"].update('Landmarks Generation ...')
            window.refresh()

            start = time.time()
            generate_landmarks(args, model_path, audio_path, actors_file, save_landmarks_path, template_name)

            print('Meshes Generation')
            window["-OUTPUT_DEMO-"].update('Meshes Generation ...')
            window.refresh()


            generate_meshes_from_landmarks(template_path, template_path, save_landmarks_path, prediction_path, save_path_meshes, args)
            end = time.time()

            print(str(end - start) + ' Seconds')

            save_video_path = save_path

            window["-OUTPUT_DEMO-"].update('Video Generation ...')
            window.refresh()
            print('Video Generation')
            generate_mesh_video(save_video_path,
                                args.video_name,
                                save_path_meshes,
                                args.fps,
                                audio_path,
                                template_path,
                                window)
            
            # Generate the video and save it
            video_path = os.path.join(save_video_path, args.video_name)
            print('done')
            media_list.add_media(video_path)
            list_player.set_media_list(media_list)
            list_player.play()
            window["-OUTPUT-"].update('Demo Completed')
            window.refresh()
            
            
        elif event == 'Play':
            list_player.play()
        elif event == 'Pause':
            list_player.pause()
        elif event == 'Exit':
            window.close()
    

if __name__ == '__main__':
    main()

