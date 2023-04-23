import numpy as np
import time
import sys

N = int(sys.argv[1])
iter = int(sys.argv[2])

half = iter//2
print(half)
A = np.ones(N,dtype=np.float32)
B = np.ones(N,dtype=np.float32)

def dp(N,A,B):
    R = np.dot(A,B)
    return R

total_time = 0

for i in range(iter):
    start = time.monotonic()
    dp(N,A,B)
    end = time.monotonic()
    # print(start,end, end-start)
    if N < 30000000 and i % 50 == 0:
        print(i)
    elif N > 30000000 and i % 4 == 0:
        print(i)
    if i>=half:
        total_time += (end-start)


avg_time = total_time / half
bw = 2 * N * np.dtype(np.float32).itemsize / avg_time / (1024 * 1024 * 1024)
flops = 2 * N / (avg_time * 1e9)
ai = flops/bw

text = "N: {} <T in usec>: {:.9f} sec B: {:.6f} GB/sec F: {:.6f} GFLOP/sec ai: {:.6f} FLOP/byte\n\n".format(N,avg_time,bw,flops,ai)

print(text)

with open('/home/hw1/dp.txt','a') as f:
    f.write("\ndp4.py\n")
    f.write(text)
