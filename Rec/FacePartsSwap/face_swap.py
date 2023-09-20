import cv2
import face_alignment
import h5py
import numpy as np
import scipy.io
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import trimesh
from Rec.FacePartsSwap.Matrix_operations import Matrix_op, Vector_op
from Rec.FacePartsSwap._3DMM import _3DMM
import scipy.io


def rec():
    # 3DMM fitting regularization
    _lambda = 0.5
    # 3DMM fitting rounds (default 1)
    _rounds = 1
    # Parameters for the HPR operator (do not change)
    _r = 3
    _C_dist = 700
    # Frontalized image rendering step: smaller -> higher resolution image
    _rendering_step = 0.5
    # Params dict
    params3dmm = {'lambda': _lambda, 'rounds': _rounds, 'r': _r, 'Cdist': _C_dist}
    # Instantiate all objects
    _3DMM_obj = _3DMM()

    # Landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
    # Load 3DMM components and weights file

    datas =  scipy.io.loadmat('./Rec/Values/VOCA_PCA_3DMM.mat')
    avgModel = datas['avgModel']
    Components = datas['Components']
    Weights = datas['Weights']
    Components_res = datas['Components_res']
    # Setup 3D objects
    m_X_obj = Matrix_op(Components, None)
    m_X_obj.reshape(Components)
    v_weights_obj = Vector_op(np.reshape(Weights, (300,1)))
    # Load 3D Model Data
    matlab =  scipy.io.loadmat('./Rec/Values/coma_landmarks.mat')
    idx_landmarks_3D = matlab['landmarks_idx'] - 1
    idx_landmarks_3D = np.reshape(np.array(idx_landmarks_3D), (1, 68))
    landmarks_3D = np.reshape(avgModel[idx_landmarks_3D], (68,3))

    #landmarks = trimesh.Trimesh(landmarks_3D)
    #landmarks.export('./Rec/Values/new_landmarks.ply')

    # 3dmm object with data
    dict3dmm = {'compObj': m_X_obj, 'weightsObj': v_weights_obj,
                'avgModel': avgModel, 'idx_landmarks_3D': idx_landmarks_3D,
                'landmarks3D': landmarks_3D}
    # Image size after resize
    im_size = 512

    #Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually your built-in webcam)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Capture a single frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        exit()

    # Save the captured frame as an image

    cv2.imwrite("./Rec/captured_image.jpg", frame)

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    #print("Image captured and saved as 'captured_image.jpg'.")

    preds = fa.get_landmarks(frame)
    preds = np.reshape(np.array(preds), (68, 3))
    preds_points = trimesh.Trimesh(preds)
    preds_points.export('./Rec/Values/land_extracted.ply')
    print('landmarks extracted')

    def deformAndResample3DMM(obj3dmm, dict3dmm, lm, params):
        estimation = obj3dmm.opt_3DMM_fast(dict3dmm['weightsObj'].V, dict3dmm['compObj'].X_after_training,
                                        dict3dmm['compObj'].X_res,
                                        dict3dmm['landmarks3D'],
                                        dict3dmm['idx_landmarks_3D'],
                                        lm,
                                        dict3dmm['avgModel'],
                                        params['lambda'],
                                        params['rounds'],
                                        params['r'],
                                        params['Cdist'])
        deformed_mesh = estimation["defShape"]
        return deformed_mesh #estimation, projected_mesh


    mesh = deformAndResample3DMM(_3DMM_obj, dict3dmm, preds[:, :2], params3dmm)
    tri = np.load('./Rec/Values/faces.npy')
    mesh_obj = trimesh.Trimesh(mesh, tri)
    mesh_obj.export('./Rec/reconstruction.ply')
    #mesh_obj2 = trimesh.Trimesh(avgModel, tri)
    #mesh_obj2.export('./Rec/template.ply')
    print('done')

