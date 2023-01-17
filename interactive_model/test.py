from ram import Ram
from datatype import DataType, Variable
import substreams as subs


# data types
act_dtype = DataType(8, 0, True)
weight_dtype = DataType(8, 0, True)
bias_dtype = DataType(32, 0, True)
scale_dtype = DataType(8, 0, True)
sum_dtype = DataType(32, 0, True)
mul_dtype = DataType(40, 0, True)
batchsize = 1

# instantiate rams
bias_ram = Ram(bias_dtype.width, 128, bias_dtype)               # 32 bit width -> 1 bank ram
weight_ram = Ram(weight_dtype.width, 8192, weight_dtype)        # 8 bit width -> 4 bank ram

input_ram = Ram(act_dtype.width, 2048, act_dtype)               # 8 bit width -> 4 bank ram
scale_ram = Ram(scale_dtype.width, 2048, scale_dtype)           # 8 bit width -> 4 bank ram
output_ram = Ram(act_dtype.width, 2048, act_dtype)              # 8 bit width -> 4 bank ram

# instantiate substreams
mul_rshift_round_clip = subs.Mul_rshift_round_clip(bias_dtype, weight_dtype, mul_dtype, act_dtype)


print("Test: mul_rshift_round_clip")
print(f"x = 64, 32bit, 0point, signed \n"
      f"y = 64, 8bit, 0point, signed\n"
      f"rshift = 5\n"
      f"output: {mul_rshift_round_clip.operate(Variable(64, 32), Variable(64, 8), Variable(5)).val}\n")
print(f"x = 64, 32bit, 0point, signed \n"
      f"y = 64, 8bit, 0point, signed\n"
      f"rshift = 6\n"
      f"output: {mul_rshift_round_clip.operate(Variable(64, 32), Variable(64, 8), Variable(6)).val}\n")
print(f"x = 64, 32bit, 0point, signed \n"
      f"y = 64, 8bit, 0point, signed\n"
      f"rshift = 3\n"
      f"output: {mul_rshift_round_clip.operate(Variable(64, 32), Variable(64, 8), Variable(7)).val}\n")


