import tensorflow as tf

class RegionPool(object):
    def __init__(self, op1, op2):
        self._op1 = op1
        self._op2 = op2
        return

    def __call__(self, inputs,x,y,w,h):
        batch, in_height, in_width, in_channels = [int(d) for d in
                                                   inputs.get_shape()]
        begin = [0, y, x, 0]
        size  = [batch, h, w, in_channels]
        t1 = tf.slice(inputs, begin, size)
        t2 = self._op1(t1)
        return self._op2(t2)

class RegionMaxPool(RegionPool):
    def __init__(self):
        def reduce_max(t):
            return tf.reduce_max(t, axis=1)

        super(self.__class__,self).__init__(reduce_max, reduce_max)
