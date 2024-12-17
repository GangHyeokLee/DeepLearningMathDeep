from functions import *

class MVLoR:
    def __init__(self, feature_dim):
        self._feature_dim = feature_dim

        self._affine = Affine_Function(feature_dim)
        self._sigmoid = Sigmoid()

    def forward(self, x):
        z = self._affine.forward(x)
        pred = self._sigmoid.forward(z)
        return pred

    def backward(self, dpred, lr):
        dz = self._sigmoid.backward(dpred)
        self._affine.backward(dz, lr)

    def get_Th(self):
        return self._affine.get_Th()