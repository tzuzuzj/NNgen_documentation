class DataType:
    def __init__(self, width=8, point=0, signed=True):
        self.width = width
        self.point = point
        self.signed = signed

    def same_type_as(self, type_to_compare):
        if self.width > type_to_compare.width or \
                self.point != type_to_compare.point or \
                self.signed != type_to_compare.signed:
            return False
        else:
            return True


class Variable(DataType):
    def __init__(self, val=0, width=8, point=0, signed=True, dType=None):
        if isinstance(dType, DataType):
            super().__init__(dType.width, dType.point, dType.signed)
        else:
            super().__init__(width, point, signed)
        self._val = val
        self.val_in_bounds()

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val
        self.val_in_bounds()

    def val_in_bounds(self):
        if not self.signed:
            if self.val < 0:
                self._val = 0
            if self.val > 2 ** self.width - 1:
                self.val = 2 ** self.width - 1
        else:
            if self.val > 2 ** (self.width - 1) - 1:
                self.val = 2 ** (self.width - 1) - 1
            if self.val < - (2 ** (self.width - 1)):
                self.val = - (2 ** (self.width - 1))


# data types
act_dtype = DataType(8, 0, True)
weight_dtype = DataType(8, 0, True)
bias_dtype = DataType(32, 0, True)
scale_dtype = DataType(8, 0, True)
sum_dtype = DataType(32, 0, True)
mul_dtype = DataType(40, 0, True)
mul_short_dtype = DataType(16, 0, True)
vshamt_mul_dtype = DataType(8, 0, False)
vshamt_sum_dtype = DataType(8, 0, False)
vshamt_out_dtype = DataType(8, 0, False)
batchsize = 1