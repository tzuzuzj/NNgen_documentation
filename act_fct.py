from datatype import *


class Relu:
    def __init__(self, out_dType, in_dType):
        self.out_dType = out_dType
        self.in_dType = in_dType

    def operate(self, x):
        if x.val > 0:
            return Variable(x.val, dType=self.out_dType)
        else:
            return Variable(0, dType=self.out_dType)
