import tensorflow as tf
from region_pool import RegionMaxPool

def max_pool_regions(data, regions):
    v = [RegionMaxPool(x,y,w,h)(data) for (x,y,w,h) in regions]
    v = tf.stack(v, axis=1)
    return v