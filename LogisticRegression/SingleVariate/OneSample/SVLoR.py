from SVLoRModules import Affine, Sigmoid


class SVLoR:
    def __init__(self):
        self._feature_dim = 1

        self._affine = Affine()
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
