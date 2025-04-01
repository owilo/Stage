import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()
    
@tf.keras.utils.register_keras_serializable()
class GradientReversal(layers.Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_
    
    def call(self, inputs):
        @tf.custom_gradient
        def reverse_grad(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse_grad(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"lambda_": self.lambda_})
        return config