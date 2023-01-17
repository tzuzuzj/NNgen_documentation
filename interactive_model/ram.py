from datatype import DataType, Variable


class Ram:
    def __init__(self, width, length, data_type):
        self.width = width
        self.length = length
        self.num_banks = int(32/width)
        self.data_type = data_type
        self._data = [[0 for i in range(self.num_banks)] for j in range(length)]

    def read(self, raddr, bank=0):
        if raddr >= self.length or raddr < 0:
            raise Exception('raddr out of bound')

        return Variable(self._data[raddr][bank], dType=self.data_type)

    def write(self, waddr, data, bank=0):
        if not data.same_type_as(self.data_type):
            raise Exception('input dataType not as expected')

        if waddr >= self.length or waddr < 0:
            raise Exception('waddr out of bound')

        self._data[waddr][bank] = data.val
