from datatype import *
from ram import Ram
import substreams as subs
from matmul_layer import Matmul_stream, Matmul_control_sequence
from act_fct import Relu
import check_computation


# address map (local and global addresses combined)
addr_map = {'output': (0, 63),                  # 64 B
            'placeholder': (64, 895),           # 832 B
            'w0': (896, 101247),                # 98 KB
            'b0': (101248, 101759),             # 512 B
            's0': (101760, 101887),             # 128 B
            'w1': (101888, 103167),             # 2 KB
            'b1': (103168, 103231),             # 64 B
            's1': (103232, 103295),             # 64 B
            'temporal': (103296, 103423)}       # 128 B

# input/output data and parameter
axi_interface = []
for i in range(addr_map['temporal'][1]+1):
    axi_interface.append(-1 ** ((i % 2) + 1) * i % 50)


# instantiate rams
bias_ram = Ram(bias_dtype.width, 128, bias_dtype)               # 32 bit width -> 1 bank ram
weight_ram = Ram(weight_dtype.width, 8192, weight_dtype)        # 8 bit width -> 4 bank ram
input_ram = Ram(act_dtype.width, 2048, act_dtype)               # 8 bit width -> 4 bank ram
scale_ram = Ram(scale_dtype.width, 2048, scale_dtype)           # 8 bit width -> 4 bank ram
output_ram = Ram(act_dtype.width, 2048, act_dtype)              # 8 bit width -> 4 bank ram


# instantiate substreams
substreams = {'mul_rshift_round_clip': subs.Mul_rshift_round_clip(bias_dtype, weight_dtype, mul_dtype, act_dtype),
              'mul_rshift_round_madd': subs.Mul_rshift_round_madd(act_dtype, weight_dtype, mul_short_dtype),
              'acc_rshift_round_frac': subs.Acc_rshift_round_frac(sum_dtype, sum_dtype),
              'add_tree': subs.Add_Tree(bias_dtype)}

# instantiate activation function
relu = Relu(act_dtype, act_dtype)

# instantiate Streams
streams = {'matmul_1': Matmul_stream(substreams, relu, bias_dtype, scale_dtype, vshamt_mul_dtype, vshamt_sum_dtype, vshamt_out_dtype,
                                     act_dtype, weight_dtype, 0, 0, 18, size=784),
           'matmul_2': Matmul_stream(substreams, relu, bias_dtype, scale_dtype, vshamt_mul_dtype, vshamt_sum_dtype, vshamt_out_dtype,
                                     act_dtype, weight_dtype, 0, 0, 15, size=128)
           }


# instantiate control sequences
matmul_l1_addr = {'output': addr_map['temporal'],           # 128 B
                  'placeholder': addr_map['placeholder'],   # 832 B
                  'w': addr_map['w0'],                      # 98 KB
                  'b': addr_map['b0'],                      # 512 B
                  's': addr_map['s0']}                      # 128 B
matmul_l1 = Matmul_control_sequence(streams['matmul_1'], matmul_l1_addr, 3136, 784, 128, 128, weight_dtype, bias_dtype,
                                    scale_dtype, act_dtype)

matmul_l2_addr = {'output': addr_map['output'],             # 64 B
                  'placeholder': addr_map['temporal'],      # 128 B
                  'w': addr_map['w1'],                      # 2 KB
                  'b': addr_map['b1'],                      # 64 B
                  's': addr_map['s1']}                      # 64 B
matmul_l2 = Matmul_control_sequence(streams['matmul_2'], matmul_l2_addr, 1280, 128, 10, 10, weight_dtype, bias_dtype,
                                    scale_dtype, act_dtype)



# ------------------
# main fsm
# ------------------

# axi configuration state 0 to 2

# ...

# compute reference result
check_computation.test(axi_interface, addr_map, 100352, 512, 128, 784, 1280, 10, 10, 128, 18, 15)

print()
print("emulation of nngen layers:")

# execute matmul layer 0 fsm
print('o0:')
matmul_l1.operate(axi_interface, bias_ram, scale_ram, weight_ram, input_ram, output_ram)

# execute matmul layer 1 fsm
print('o1:')
matmul_l2.operate(axi_interface, bias_ram, scale_ram, weight_ram, input_ram, output_ram)
