import tensorflow as tf
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

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
class RealNVPFlow(layers.Layer):
    def __init__(self, num_coupling_layers: int, hidden_units=(128,128), **kwargs):
        super().__init__(**kwargs)
        self.num_coupling = num_coupling_layers
        self.hidden_units = hidden_units
        self._nets = []          
        self._bijectors = []

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self._nets.clear()
        self._bijectors.clear()

        for i in range(self.num_coupling):
            net = tf.keras.Sequential(name=f"rnvpf_net_{i}", layers=[
                layers.Dense(units, activation="relu", name=f"dense_{i}_{j}")
                for j, units in enumerate(self.hidden_units)
            ] + [
                layers.Dense(2 * (dim // 2), name=f"shift_log_{i}")
            ])
            self._nets.append(net)

            def make_s_and_ls(net):
                def _s_and_ls(x, *_):
                    out = net(x)
                    return tf.split(out, 2, axis=-1)
                return _s_and_ls

            bij = tfp.bijectors.RealNVP(
                num_masked=dim // 2,
                shift_and_log_scale_fn=make_s_and_ls(net),
            )
            self._bijectors.append(bij)

            with tf.init_scope():
                perm = list(reversed(range(dim)))
                self._bijectors.append(tfp.bijectors.Permute(permutation=perm))

        self.chain = tfp.bijectors.Chain(self._bijectors[::-1])
        super().build(input_shape)

    def call(self, inputs):
        return self.chain.forward(inputs)

    def inverse(self, inputs):
        return self.chain.inverse(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_coupling_layers": self.num_coupling,
            "hidden_units": self.hidden_units
        })
        return cfg

@tf.keras.utils.register_keras_serializable()
class IAFFlow(layers.Layer):
    def __init__(self,
                 hidden_units=(64, 64),
                 num_maf_layers=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden_units
        self.num_layers = num_maf_layers
        self.bijectors = []

    def build(self, input_shape):
        dim = int(input_shape[-1])
        for _ in range(self.num_layers):
            self.bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=list(self.hidden)
                    )
                )
            )
            self.bijectors.append(tfb.Permute(permutation=list(reversed(range(dim)))))
        self.chain = tfb.Chain(self.bijectors[::-1])
        super().build(input_shape)

    def call(self, inputs):
        return self.chain.forward(inputs)

    def inverse(self, inputs):
        return self.chain.inverse(inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_units": self.hidden,
            "num_maf_layers": self.num_layers,
        }

@tf.keras.utils.register_keras_serializable()
class SamplingFlow(layers.Layer):
    def __init__(self, flow_layer: layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.flow = flow_layer

    def build(self, input_shape):
        # input_shape: tuple of ([batch, dim], [batch, dim])
        z_shape = input_shape[0]
        self.flow.build(z_shape)
        super().build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # base gaussian distribution
        base = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.exp(0.5 * z_log_var))
        # transformed via flow
        flow_dist = tfd.TransformedDistribution(distribution=base, bijector=self.flow.chain)
        z = flow_dist.sample()
        log_qz = flow_dist.log_prob(z)
        return z, log_qz

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"flow_layer": tf.keras.utils.serialize_keras_object(self.flow)})
        return cfg

    @classmethod
    def from_config(cls, config):
        flow_config = config.pop("flow_layer")
        flow = tf.keras.utils.deserialize_keras_object(flow_config)
        return cls(flow, **config)
    
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