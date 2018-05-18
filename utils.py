import tensorflow as tf
from region_pool import RegionMaxPool

def pool_region_array(pooler, data, ar):
    v = [pooler(data,x,y,w,h) for (x,y,w,h) in ar]
    return tf.stack(v, axis=1)


def max_pool_regions(data, regions):
    pooler = RegionMaxPool()
    v = [pool_region_array(pooler, data, ar) for ar in regions]
    v = tf.stack(v, axis=1)
    return v
