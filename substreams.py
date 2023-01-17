import xml.dom.minidom

from datatype import DataType, Variable
import math

import veriloggen.stream.stypes as stypes


class Mul_rshift_round_clip:
    '''
    computation module (sub stream) that multiplies x and y, rightshifts the result by rshift and clips the output values
    to prevent overflows.
    '''
    def __init__(self, x_dType, y_dType, mul_dType, out_dType, asymmetric_clip=False):
        # internal
        self.internal_dType = DataType(max(x_dType.width, y_dType.width), max(x_dType.point, y_dType.point))
        self.z_dType = mul_dType
        self.out_dType = out_dType
        self.asymmetric_clip = asymmetric_clip

        # source:
        self.x_dType = x_dType
        self.y_dType = y_dType
        self.rshift_dType = DataType(int(math.ceil(math.log(self.internal_dType.width, 2))) + 1,
                                     0, False)

    def operate(self, x, y, rshift):
        # check for correct input type
        if not x.same_type_as(self.x_dType):
            raise Exception('[Mul_rshift_round_clip]: input dataType not as expected')
        if not y.same_type_as(self.y_dType):
            raise Exception('[Mul_rshift_round_clip]: input dataType not as expected')

        rshift = Variable(rshift, dType=self.rshift_dType)
        z = Variable(dType=self.z_dType)
        z.val = x.val * y.val
        z = sra_round(z, rshift)

        # clipping
        p_th = (1 << (self.out_dType.width - 1)) - 1
        if self.asymmetric_clip:
            n_th = -1 * p_th - 1
        else:
            n_th = -1 * p_th

        p_th = p_th >> self.out_dType.point
        n_th = n_th >> self.out_dType.point

        if z.val > p_th:
            z.val = p_th
        elif z.val < n_th:
            z.val = n_th

        return Variable(z.val, dType=self.out_dType)


class Mul_rshift_round_madd:
    '''
    x >= 0:
        return (x * y + (1 << (rshift - 1))) >> rshift
    else
        return (x * y - (1 << (rshift - 1))) >> rshift
    '''
    def __init__(self, x_dType, y_dType, mul_dType):
        # internal:
        self.internal_dType = DataType(max(x_dType.width, y_dType.width), max(x_dType.point, y_dType.point))
        self.mul_dType = mul_dType

        # source:
        self.x_dType = x_dType
        self.y_dType = y_dType
        self.rshift_dType = DataType(int(math.ceil(math.log(self.internal_dType.width, 2))) + 1,
                                     0, False)

    def operate(self, x, y, rshift):
        if not x.same_type_as(self.x_dType):
            raise Exception('[Mul_rshift_round_madd]: input dataType not as expected')
        if not y.same_type_as(self.y_dType):
            raise Exception('[Mul_rshift_round_madd]: input dataType not as expected')

        rshift = Variable(rshift, dType=self.rshift_dType)
        frac = Variable(dType=self.mul_dType)

        if rshift.val > 0:
            frac.val = sll(1, rshift.val - 1)
        else:
            frac.val = 0

        if x.val < 0:
            frac = uminus(frac)

        frac_shorter = Variable(frac.val, self.internal_dType.width, self.mul_dType.point, self.mul_dType.signed)

        z = mulAdd(x, y, frac_shorter)
        z.val = sra(z.val, rshift.val)

        return z


class Add_Tree:
    '''
    does nothing when par = 1!?
    '''
    def __init__(self, x_dType):
        self.x_dType = x_dType

    def operate(self, x):
        if type(x) != list:
            sum = x
            return sum
        else:
            # sum up inputs ... not implemented
            return x


class Acc_rshift_round_frac:
    '''
    used for conv2d and matmul
    '''

    def __init__(self, x_dType, sum_dType):
        self.x_dType = x_dType
        self.sum_dType = sum_dType
        self.rshift_dType = DataType(int(math.ceil(math.log(x_dType.width, 2))) + 1)
        self._sum = Variable(0, dType=self.sum_dType)
        self._cnt = 0

    def operate(self, x, rshift, size):
        if not x.same_type_as(self.x_dType):
            raise Exception('[Mul_rshift_round_madd]: input dataType not as expected')

        rshift = Variable(rshift, dType=self.rshift_dType)
        frac = Variable(0, dType=self.sum_dType)

        if rshift.val > 0:
            frac.val = sll(1, rshift.val - 1)

        self._sum.val += x.val              # ReduceAddValid... accumulate the input values
        self._cnt += 1

        if self._cnt >= size:
            self._sum.val += frac.val
            self._sum.val = sra(self._sum.val, rshift.val)
            tmp = Variable(self._sum.val, dType=self.sum_dType)
            self._sum.val = 0               # reset sum
            self._cnt = 0                   # reset counter
            valid = True
            return tmp, valid
        else:
            valid = False
            return self._sum, valid


# --------------------------------------------------------------------------------
# -------------------------------- Stream methodes -------------------------------
# --------------------------------------------------------------------------------


def sra_round(left, right):
    # assumption: right is always int
    if isinstance(right.val, int) and right.val > 0:
        rounder = 1 << right.val - 1
    else:
        raise Exception("[sra_round]: negative shifts and floating shifts not implemented yet")

    # check if left.msb = 1
    if left.val >= 2 ** (left.width - 1) or left.val < 0:
        rounder_sign = -1
    else:
        rounder_sign = 0

    pre_round = Variable(left.val + rounder + rounder_sign, left.width + 1, left.point, left.signed)
    shifted = Variable(sra(pre_round.val, right.val), left.width, left.point, left.signed)

    return shifted


def uminus(x):
    '''
    negates value of variable x
    '''
    return Variable(- x.val, x.width, x.point, x.signed)


def mulAdd(a, b, c):
    '''
    returns a * b + c
    same as Madd, mkMaddCore
    '''
    return Variable(a.val * b.val + c.val, max(a.width + b.width, c.width), 0, True)


def sll(left, right):
    if left < 0:
        raise Exception("[sll]: Arithmetic shift instead of logical shift due to negative value")

    return left << right


def sra(left, right):
    return left >> right

