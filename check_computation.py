import math

def test (axi_interface, addr_map, w0_size, b0_size, s0_size, a0_size, w1_size, b1_size, s1_size, a1_size, o0_rshift, o1_rshift):
    print("reference values: ")

    w0 = axi_interface[addr_map['w0'][0]:addr_map['w0'][0]+w0_size]
    b0 = axi_interface[addr_map['b0'][0]:addr_map['b0'][0]+b0_size*4]
    s0 = axi_interface[addr_map['s0'][0]:addr_map['s0'][0]+s0_size]
    a0 = axi_interface[addr_map['placeholder'][0]:addr_map['placeholder'][0]+a0_size]
    o0 = computation(w0, b0, s0, a0, o0_rshift)

    w1 = axi_interface[addr_map['w1'][0]:addr_map['w1'][0]+w1_size]
    b1 = axi_interface[addr_map['b1'][0]:addr_map['b1'][0]+b1_size*4]
    s1 = axi_interface[addr_map['s1'][0]:addr_map['s1'][0]+s1_size]
    a1 = o0
    o1 = computation(w1, b1, s1, a1, o1_rshift)

    print("o0:")
    print(o0)
    print("o1:")
    print(o1)


def computation (weights, biases, scales, acts, rshift):
    o = []

    for i in range(len(scales)):
        s = scales[i]
        b = biases[i*4]       # 32 bit values
        acc = 0
        for j in range(len(acts)):
            a = acts[j]
            # print('i:', i, 'j:', j)
            w = weights[len(acts) * i + j]

            acc += a * w
        out = (int(round(((acc + b) * s) / 2**rshift, 0)))      # right shift by 18/15 digits
        o.append(max(out, 0))                                   # relu
    return o
