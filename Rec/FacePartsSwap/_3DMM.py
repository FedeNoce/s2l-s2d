import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from scipy.spatial import ConvexHull


class _3DMM:
    result = {}  # dictionary to return

    def opt_3DMM_fast(self, Weights, Components, Components_res, landmarks3D, id_landmarks_3D, landImage, avgFace, _lambda, rounds, r, C_dist):
        _Weights = Weights
        index = np.array(id_landmarks_3D, dtype=np.intp)
        app_var = np.reshape(avgFace[index,:],(landmarks3D.shape[0],3),order='F')
        # _var --> variabili di appoggio temporanee
        [_Aa, _Sa, _Ra, _Ta] = self.estimate_pose(app_var, landImage)
        _proj = np.transpose(self.getProjectedVertex(app_var, _Sa, _Ra, _Ta))
        # deform shape
        _alpha = self.alphaEstimation(landImage, _proj, Components_res, id_landmarks_3D, _Sa, _Ra, _Ta, Weights, _lambda)
        _defShape = self.deform_3D_shape_fast(np.transpose(avgFace), Components, _alpha)
        _defShape = np.transpose(_defShape)
        _defLand = np.reshape(_defShape[index,:],(landmarks3D.shape[0],3),order='F')
        _proj = np.transpose(self.getProjectedVertex(_defLand, _Sa, _Ra, _Ta))
        for i in range(rounds):
            [_Aa, _Sa, _Ra, _Ta] = self.estimate_pose(_defLand, landImage)
            _proj = np.transpose(self.getProjectedVertex(_defLand, _Sa, _Ra, _Ta))
            _alpha = self.alphaEstimation(landImage, _proj, Components_res, id_landmarks_3D, _Sa, _Ra, _Ta, Weights,_lambda)
            _defShape = self.deform_3D_shape_fast(np.transpose(_defShape),Components,_alpha)
            _defShape = np.transpose(_defShape)
            _defLand = np.reshape(_defShape[index, :], (landmarks3D.shape[0], 3), order='F')

        [_Aa, _Sa, _Ra, _Ta] = self.estimate_pose(_defLand, landImage)
        _proj = np.transpose(self.getProjectedVertex(_defLand, _Sa, _Ra, _Ta))
        _visIdx = self.estimateVis_vertex(_defShape, _Ra, C_dist, r)
        _visIdx = np.delete(_visIdx, (0), axis=0)

        # assing values to dictionary
        self.result["A"] = _Aa
        self.result["S"] = _Sa
        self.result["R"] = _Ra
        self.result["T"] = _Ta
        self.result["defShape"] = _defShape
        self.result["alpha"] = _alpha
        self.result["visIdx"] = _visIdx

        return self.result

    def estimate_pose(self, landModel, landImage):
        baricM = np.mean(landModel,axis=0)
        P = landModel - np.tile(baricM,(landModel.shape[0],1))

        baricI = np.mean(landImage,axis=0)
        p = landImage - np.tile(baricI,(landImage.shape[0],1))

        P = np.transpose(P)
        p = np.transpose(p)
        qbar = np.transpose(baricI)
        Qbar = np.transpose(baricM)
        A = p.dot(pinv(P))
        [S,R] = np.linalg.qr(np.transpose(A), mode='complete') # A = S*R
        rr = S
        S = np.transpose(R)
        R = np.transpose(rr)
        t = qbar - A.dot(Qbar)
        # save data to the dictionary
        return [A,S,R,t]

    def getProjectedVertex(self,vertex,S,R,t):
        _vertex = self.transVertex(vertex)
        rotPc = np.transpose(R.dot(_vertex))
        t = np.reshape(t,(2,1),order='F')
        return S.dot(np.transpose(rotPc)) + np.tile(t, (1,rotPc.shape[0]))

    def transVertex(self,vertex):
        v = vertex
        if vertex.shape[0] != 3:
            return np.transpose(v)
        else:
            return v

    def alphaEstimation(self, landImage, projLandModel, Components_res, id_landmarks_3D, S, R, t, Weights, _lambda):
        Weights = np.reshape(Weights, (Weights.shape[0], 1))
        X = landImage - projLandModel
        X = X.flatten(order='F')
        Y = np.zeros((X.shape[0],Components_res.shape[2]))
        index = np.array(id_landmarks_3D,dtype=np.intp)

        for c in range(Components_res.shape[2]):
            vect = Components_res[index,:,c].reshape(landImage.shape[0],3,order='F')
            vertexOnImComp = np.transpose(self.getProjectedVertex(vect,S,R,t))
            Y[:,c] = vertexOnImComp.flatten(order='F')
        if _lambda == 0:
            Alpha = Y.divide(X)
        else:
            with np.errstate(divide='ignore'):
                invW = np.diag( _lambda/(np.diagflat(Weights))) # as diag in matlab
            var = (np.transpose(Y)).dot(Y)
            res = var + np.diagflat(invW)
            YY = inv(res)
            app = np.dot(YY, np.transpose(Y))
            Alpha = np.dot(app,X)
        return Alpha



    def deform_3D_shape_fast(self, mean_face, eigenvecs, alpha):
        dim = (eigenvecs.shape[0])//3
        alpha_full = np.tile(np.transpose(alpha), (eigenvecs.shape[0],1))
        tmp_eigen = alpha_full*eigenvecs
        sumVec = np.sum(tmp_eigen, axis=1)  
        sumMat = np.reshape(np.transpose(sumVec), (3, dim), order='F')
        return mean_face + sumMat


    def estimateVis_vertex(self, vertex, R, C_dist, r):
        viewPoint_front = np.array([0,0,C_dist]).reshape(1,3, order='F')
        viewPoint = np.transpose(self.rotatePointCloud(viewPoint_front, R, []))
        visIdx = self.HPR(vertex, viewPoint, r)  # controllare la funzione HPR

        return visIdx

    def rotatePointCloud(self, P, R, t):        
        return np.dot(R, np.transpose(P))

    def HPR(self, p, C, param):
        dim = p.shape[1]
        numPts = p.shape[0]
        p = p - np.tile(C,(numPts,1))
        #normP = np.sqrt(p.dot(p))
        normP = np.linalg.norm(p, axis=1)
        normP = normP.reshape(normP.shape[0], 1, order='F')
        app = np.amax(normP)*(np.power(10,param))
        R = np.tile(app,(numPts,1))

        P = p + 2*np.tile((R-normP),(1,dim))*p/np.tile(normP,(1,dim))
        _zeros = np.zeros((1,dim))
        vect_conv_hull = np.vstack([P,_zeros])
        hull = ConvexHull(vect_conv_hull)
        visiblePtInds = np.unique(hull.vertices)
        for i in range(visiblePtInds.shape[0]):
            visiblePtInds[i] =  visiblePtInds[i] - 1
            if visiblePtInds[i] == (numPts + 1):
                visiblePtInds.remove(i)

        return visiblePtInds.reshape(visiblePtInds.shape[0], 1, order='F')

    def getVisLand(self, vertex, landImage, visIdx, id_landmarks_3D):
        visLand = np.intersect1d(visIdx, id_landmarks_3D)
        mb = np.in1d(id_landmarks_3D, visIdx)
        id_Vland = np.nonzero(mb)[0]

        app = np.empty([1,landImage.shape[0]])
        for i in range(1,landImage.shape[0]):
            app[0,i] = i

        Nvis = np.setxor1d(app,id_Vland)
        return [visLand, id_Vland, Nvis]

