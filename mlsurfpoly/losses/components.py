from __future__ import annotations

import tensorflow as tf


def weighted_bce_loss(y_true, y_pred, pos_weight: float = 0.5):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    loss = -(
        pos_weight * y_true * tf.math.log(y_pred)
        + (1.0 - y_true) * tf.math.log(1 - y_pred)
    )
    return tf.reduce_mean(loss)


def sequence_prediction_loss(y_true, y_pred):
    surface_true = y_true[:, :400]
    surface_pred = y_pred[:, :400]
    polymer_true = y_true[:, 400:]
    polymer_pred = y_pred[:, 400:]
    surface_bce = weighted_bce_loss(surface_true, surface_pred, pos_weight=0.5)
    polymer_bce = tf.keras.losses.binary_crossentropy(polymer_true, polymer_pred)
    return 0.9 * surface_bce + 0.01 * tf.reduce_mean(polymer_bce)


def pmf_reconstruction_loss(x_true, y_pred):
    x_true = tf.cast(x_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(x_true - y_pred))


def pmf_prediction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def bcext_reconstruction_loss(y_pred, y_true):
    return tf.reduce_mean(tf.abs(y_true[:, 1:] - y_pred[:, 1:]))


def backext_loss(y_pred_tuple, y_true_tuple):
    pmf_pred, deltaF_pred = y_pred_tuple
    pmf_true, deltaF_true = y_true_tuple
    pmf_loss = tf.reduce_mean(tf.abs(pmf_true - pmf_pred))
    deltaF_loss = tf.reduce_mean(tf.abs(deltaF_true - deltaF_pred))
    return pmf_loss, deltaF_loss
