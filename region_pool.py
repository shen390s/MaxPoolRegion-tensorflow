import tensorflow as tf

class RegionMaxPool(object):
    def __init__(self):
        return

    def __call__(self, inputs,x,y,w,h):
        batch, in_height, in_width, in_channels = [int(d) for d in
                                                   inputs.get_shape()]
        begin = [0, y, x, 0]
        size  = [batch, h, w, in_channels]
        t1 = tf.slice(inputs, begin, size)
        t2 = tf.reduce_max(t1, axis=1)
        return tf.reduce_max(t2, axis=1)
