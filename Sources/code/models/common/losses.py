import tensorflow as tf

def reconstruction_loss(true, pred, reduction_axes=(1,2)):
    bce = tf.keras.losses.binary_crossentropy(true, pred)
    return tf.reduce_mean(tf.reduce_sum(bce, axis=reduction_axes))

def kl_divergence(z_mean, z_log_var):
    return -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    )

def vae_loss(true, pred, z_mean, z_log_var, beta=1.0):
    recon = reconstruction_loss_fn(true, pred)
    kl = tf.reduce_mean( kl_divergence(z_mean, z_log_var) )
    return recon + beta * kl, recon, beta * kl