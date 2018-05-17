from __future__ import print_function
import random
import sys
import numpy as np
import tensorflow as tf
from region_pool import RegionMaxPool
from utils import max_pool_regions

def gen_data(b,h,w,c):
    return np.random.random((b,h,w,c))

def print_data(data):
    b,h,w,c = data.shape
    for i in range(b):
        for j in range(c):
            for y in range(h):
                for x in range(w):
                    print(data[i,y,x,j],end=' ')
                print("\n")
            print("\n")


d = gen_data(2,6,7,3)

print("data:\n")
print_data(d)

rp = RegionMaxPool(2,2,3,3)

print("rp = ", rp)

rx = rp(tf.constant(d))

print("rx = ", rx)

v2 = max_pool_regions(tf.constant(d), [(1,2,2,3), (2,2,3,3)])
print("v2=", v2)

sess = tf.Session()

v = sess.run(rx)

print("v=", v)

print("v2=", sess.run(v2))
