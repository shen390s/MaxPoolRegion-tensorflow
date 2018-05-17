import tensorflow as tf

class RegionMaxPool(object):
    def __init__(self,x,y,w,h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def __call__(self, inputs, name=None):
        batch, in_height, in_width, in_channels = [int(d) for d in
                                                   inputs.get_shape()]
        begin = [0, self._y, self._x, 0]
        size  = [batch, self._h, self._w, in_channels]
        t1 = tf.slice(inputs, begin, size, name)
        t2 = tf.reduce_max(t1, axis=1)
        return tf.reduce_max(t2, axis=1)
