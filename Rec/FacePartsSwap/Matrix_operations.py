import numpy as np
from numpy.linalg import norm

class Matrix_op:
    X_aligned_model = []
    X_after_training = []
    X_res = []
    def __init__(self, Components, aligned_models_data):
        if aligned_models_data is None:
            self.X_after_training = Components
            self.reshape(Components)
            X_aligned_model = None
        if Components is None:
            self.X_aligned_model = np.array(aligned_models_data)  # 2-d Dimension and saved in memory with the fortran memory order
            self.X_after_training = []
            self.X_res = []

    def mean(self):
        self.X_aligned_model = self.X_aligned_model - self.X_aligned_model.mean(axis=1, keepdims=True) # axis=1 for get row's mean (axis=0 for columns)

    def normalization(self):
        self.X_aligned_model = norm(self.X_aligned_model, axis=1,ord=1) #L1-norm

    def transpose(self):
        self.X_aligned_model = np.transpose(self.X_aligned_model)

    def reshape(self,D) -> object:
        self.X_res = np.empty((int(D.shape[0]/3),3,D.shape[1]))
        for c in range(D.shape[1]):
            comp = np.transpose(np.array(D[:,c]))
            _app = np.reshape(np.transpose(comp),(3,int(D.shape[0]/3)), order='F')
            comp = np.transpose(_app)
            self.X_res[:,:,c] = comp

class Vector_op:
    V = []
    def __init__(self,v_init):
        self.V = v_init

    def scale(self,mx,mn):
        min_w = np.amin(self.V)
        max_w = np.amax(self.V)
        self.V = (((self.V-min_w)*(mx-mn))/(max_w-min_w)) + mn
