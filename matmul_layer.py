import substreams as subs
from datatype import *

class Matmul_stream:
    '''
    represents nngen/operator/conv2d.py/get_stream_func
    line: 1011
    '''

    def __init__(self, substreams, act_fct, bias_dType, scale_dType, vshamt_mul_dType, vshamt_sum_dType, vshamt_out_dType,
                 act_var_dType, filter_var_dType, cshamt_mul, cshamt_sum, cshamt_out, size):

        self.substreams = substreams
        self.act_fct = act_fct

        # input types
        self.bias_dType = bias_dType
        self.scale_dType = scale_dType
        self.vshamt_mul_dType = vshamt_mul_dType
        self.vshamt_sum_dType = vshamt_sum_dType
        self.vshamt_out_dType = vshamt_out_dType
        self.act_var_dType = act_var_dType
        self.filter_var_dType = filter_var_dType

        # constants
        self.cshamt_mul = cshamt_mul
        self.cshamt_sum = cshamt_sum
        self.cshamt_out = cshamt_out
        self.size = size

    def operate(self, vec_bias, vec_scale, vec_vshamt_mul, vec_vshamt_sum, vec_vshamt_out, vec_act_var, vec_filter_var):
        '''
        out = (((act_var * filter_var) + bias) * scale) >> vshamt_out
        '''

        # assumption: par = 1
        bias = vec_bias
        scale = vec_scale
        vshamt_mul = vec_vshamt_mul
        vshamt_sum = vec_vshamt_sum
        vshamt_out = vec_vshamt_out
        act_var = vec_act_var
        filter_var = vec_filter_var

        # check for correct input types
        if not bias.same_type_as(self.bias_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')
        if not scale.same_type_as(self.scale_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')
        if not vshamt_mul.same_type_as(self.vshamt_mul_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')
        if not vshamt_sum.same_type_as(self.vshamt_sum_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')
        if not vshamt_out.same_type_as(self.vshamt_out_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')
        if not act_var.same_type_as(self.act_var_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')
        if not filter_var.same_type_as(self.filter_var_dType):
            raise Exception('[' + __name__ + ']: input dataType not as expected')

        _res = self.substreams['mul_rshift_round_madd'].operate(act_var, filter_var, vshamt_mul.val + self.cshamt_mul)
        _res = self.substreams['add_tree'].operate(_res)
        _res, val = self.substreams['acc_rshift_round_frac'].operate(_res, vshamt_sum.val + self.cshamt_sum, self.size)

        if val:
            _res.val += bias.val
            _res = self.substreams['mul_rshift_round_clip'].operate(_res, scale, vshamt_out.val + self.cshamt_out)
            _res = self.act_fct.operate(_res)
            return _res, True

        return _res, False


class Matmul_control_sequence:
    '''
    represents nngen/operator/conv2d.py/control_sequence
    line 1766
    and .../get_control_param_values
    line 1378
    '''

    def __init__(self, stream, addresses,
                 filter_read_size, act_read_size, bias_read_size, scale_read_size,
                 w_dtype, b_dtype, s_dtype, act_dtype):

        self.stream = stream

        self.bias_addr = addresses['b']
        self.bias_read_size = bias_read_size        # 128
        self.b_dtype = b_dtype
        self.bias_read_size_bytes = bias_read_size * b_dtype.width // 8

        self.scale_addr = addresses['s']
        self.scale_read_size = scale_read_size      # 128, 10
        self.s_dtype = s_dtype
        self.scale_read_size_bytes = scale_read_size * s_dtype.width // 8

        # 784 input nodes fully connected to 128 output nodes correspond to 100352 weights.
        # with a filter_read_size of 3136 we compute 4 output nodes per iteration.
        self.filter_addr = addresses['w']
        self.filter_read_size = filter_read_size    # 3136
        self.w_dtype = w_dtype
        self.filter_read_size_bytes = filter_read_size * w_dtype.width // 8

        self.act_addr = addresses['placeholder']
        self.act_read_size = act_read_size          # 784, 128
        self.act_dtype = act_dtype
        self.act_read_size_bytes = act_read_size * act_dtype.width // 8

        self.out_addr = addresses['output']
        self.out_dtype = act_dtype

    def operate(self, axi_interface, bias_ram, scale_ram, filter_ram, input_ram, output_ram):

        # -----------
        # init phase
        # -----------

        # dma read - bias
        for addr in range(self.bias_read_size):
            addr_bytes = addr * self.b_dtype.width // 8
            bias_ram.write(addr, Variable(axi_interface[self.bias_addr[0] + addr_bytes], dType=self.b_dtype))

        # dma read - scale
        for addr in range(self.scale_read_size):
            addr_bytes = addr * self.s_dtype.width // 8
            scale_ram.write(addr, Variable(axi_interface[self.scale_addr[0] + addr_bytes], dType=self.s_dtype))

        # dma read - act
        for addr in range(self.act_read_size):
            addr_bytes = addr * self.act_dtype.width // 8
            input_ram.write(addr, Variable(axi_interface[self.act_addr[0] + addr_bytes], dType=self.act_dtype))

        # -----------
        # comp / write output / update for next iteration phase (merged to for loop instead of fsm)
        # -----------

        # set stream constants

        # execute stream...
        # parse data to stream via source pattern.
        # pattern: Because we only load a fraction of the weights, we can only compute x output values this iteration.
        # The pattern describes this value x and the total number of inputs.
        bias_addr = 0
        scale_addr = 0
        out_addr = 0
        filter_local_offset = 0
        out_local_offset = 0
        next_out_write_size = self.filter_read_size//self.act_read_size
        res = 0
        valid = False
        output_list = []

        while bias_addr < self.bias_read_size:                                          # iterate over outputs

            # load next weights
            for addr in range(self.filter_read_size):
                addr_bytes = (filter_local_offset + addr) * self.w_dtype.width // 8
                filter_ram.write(addr, Variable(axi_interface[self.filter_addr[0] + addr_bytes], dType=self.w_dtype))
            filter_local_offset += self.filter_read_size
            filter_addr = 0

            for pattern in range(next_out_write_size):
                for act_addr in range(self.act_read_size):                              # act_read_size = mumber of inputs
                    bias = bias_ram.read(bias_addr)
                    scale = scale_ram.read(scale_addr)
                    weight = filter_ram.read(filter_addr)
                    act = input_ram.read(act_addr)
                    vshamt_mul = Variable(0, dType=vshamt_mul_dtype)
                    vshamt_sum = Variable(0, dType=vshamt_sum_dtype)
                    vshamt_out = Variable(0, dType=vshamt_out_dtype)

                    res, valid = self.stream.operate(bias, scale, vshamt_mul, vshamt_sum, vshamt_out, act, weight)

                    filter_addr += 1
                bias_addr += 1
                scale_addr += 1

                if valid:
                    output_ram.write(out_addr * self.out_dtype.width // 8, res)      # store output to ram
                    output_list.append(res.val)
                    valid = False
                else:
                    raise Exception('output not valid')
                out_addr += 1

            # DMA write computed output nodes
            for addr in range(next_out_write_size):
                addr_bytes = (out_local_offset + addr) * self.act_dtype.width // 8
                axi_interface[self.out_addr[0] + addr_bytes] = output_ram.read(addr).val
            out_local_offset += next_out_write_size
            out_addr = 0

        print(output_list)
