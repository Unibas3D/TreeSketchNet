import tensorflow as tf

class OneMinusRMSE(tf.keras.metrics.RootMeanSquaredError):
    def __init__(self, name='one_minus_rmse', dtype = None):
        super(OneMinusRMSE, self).__init__(name=name, dtype=dtype)

    def result(self):
        result = super(OneMinusRMSE, self).result()
        return (1.0 - result)*100.0
