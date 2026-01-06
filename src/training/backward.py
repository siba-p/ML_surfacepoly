"""Training loops for backward, tandem, and extended models."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .history import CustomHistory
from ..losses import (
    weighted_bce_loss,
    pmf_reconstruction_loss,
    backext_loss,
)


def train_backward_model(
    backward_model: keras.Model,
    forward_model: keras.Model,
    *,
    train_seq,
    train_pmf,
    valid_seq,
    valid_pmf,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    callbacks_list: Iterable[keras.callbacks.Callback] | None = None,
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history = CustomHistory([
        "trn_loss",
        "trn_surface_bce",
        "trn_pmf_recon_mae",
    ], [
        "val_loss",
        "val_surface_bce",
        "val_pmf_recon_mae",
    ])

    callbacks = tf.keras.callbacks.CallbackList(callbacks_list or [], add_history=True, model=backward_model)
    callbacks.on_train_begin()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_seq, train_pmf)).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_seq, valid_pmf)).batch(batch_size)

    for epoch in range(epochs):
        callbacks.on_epoch_begin(epoch)
        train_metrics = []
        for batch_seq, batch_pmf in train_dataset:
            batch_seq = tf.cast(batch_seq, tf.float32)
            batch_pmf = tf.cast(batch_pmf, tf.float32)
            true_surface = batch_seq[:, :400]
            polymer = batch_seq[:, 400:]

            with tf.GradientTape() as tape:
                pred_surface = backward_model(batch_pmf, training=True)
                forward_input = tf.concat([pred_surface[:, :400], polymer], axis=1)
                pred_pmf = forward_model(forward_input, training=False)
                loss_surf = weighted_bce_loss(true_surface, pred_surface[:, :400])
                loss_recon = pmf_reconstruction_loss(batch_pmf, pred_pmf)
                total_loss = loss_surf + loss_recon

            grads = tape.gradient(total_loss, backward_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, backward_model.trainable_variables))
            train_metrics.append((total_loss, loss_surf, loss_recon))

        trn_loss, trn_surface, trn_recon = map(
            lambda vals: float(np.mean(vals)),
            zip(*[(t.numpy(), s.numpy(), r.numpy()) for t, s, r in train_metrics]),
        )

        val_surface_list, val_recon_list = [], []
        for batch_seq, batch_pmf in valid_dataset:
            batch_seq = tf.cast(batch_seq, tf.float32)
            batch_pmf = tf.cast(batch_pmf, tf.float32)
            true_surface = batch_seq[:, :400]
            polymer = batch_seq[:, 400:]
            pred_surface = backward_model(batch_pmf, training=False)
            forward_input = tf.concat([pred_surface[:, :400], polymer], axis=1)
            pred_pmf = forward_model(forward_input, training=False)
            loss_surf = weighted_bce_loss(true_surface, pred_surface[:, :400])
            loss_recon = pmf_reconstruction_loss(batch_pmf, pred_pmf)
            val_surface_list.append(loss_surf.numpy())
            val_recon_list.append(loss_recon.numpy())

        val_surface = float(np.mean(val_surface_list))
        val_recon = float(np.mean(val_recon_list))
        val_loss = val_surface + val_recon

        history.update(
            [trn_loss, trn_surface, trn_recon],
            [val_loss, val_surface, val_recon],
        )

        logs = {
            "trn_loss": trn_loss,
            "trn_surface_bce": trn_surface,
            "trn_pmf_recon_mae": trn_recon,
            "val_loss": val_loss,
            "val_surface_bce": val_surface,
            "val_pmf_recon_mae": val_recon,
        }
        callbacks.on_epoch_end(epoch, logs=logs)

    callbacks.on_train_end()
    return history


def train_tandem_and_extend_model(
    extend_model: keras.Model,
    backext_model: keras.Model,
    backward_model: keras.Model,
    forward_model: keras.Model,
    *,
    sequence_train_data,
    pmf_train_data,
    extra_train_data,
    sequence_valid_data,
    pmf_valid_data,
    extra_valid_data,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    callbacks_list: Iterable[keras.callbacks.Callback] | None = None,
):
    k1, k2, k3, k4 = 0.45, 2.0, 2.0, 0.1

    sequence_train_data = tf.convert_to_tensor(sequence_train_data, dtype=tf.float32)
    pmf_train_data = tf.convert_to_tensor(pmf_train_data, dtype=tf.float32)
    extra_train_data = tf.convert_to_tensor(extra_train_data, dtype=tf.float32)

    sequence_valid_data = tf.convert_to_tensor(sequence_valid_data, dtype=tf.float32)
    pmf_valid_data = tf.convert_to_tensor(pmf_valid_data, dtype=tf.float32)
    extra_valid_data = tf.convert_to_tensor(extra_valid_data, dtype=tf.float32)

    callbacks = tf.keras.callbacks.CallbackList(callbacks_list or [], add_history=True, model=extend_model)
    callbacks.on_train_begin()

    extend_model.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    extend_model.compile(optimizer=extend_model.optimizer, loss="mae", metrics=["mse"])

    history = CustomHistory(
        ["trn_loss", "trn_pmf_backext_mae", "trn_surface_bce", "trn_pmf_recon_mae"],
        ["val_loss", "val_pmf_backext_mae", "val_surface_bce", "val_pmf_recon_mae"],
    )

    train_size = extra_train_data.shape[0]
    num_batches = int(np.ceil(train_size / batch_size))

    @tf.function
    def process_batch(batch_sequence, batch_pmf, batch_extra):
        batch_sequence = tf.cast(batch_sequence, tf.float32)
        batch_pmf = tf.cast(batch_pmf, tf.float32)
        batch_extra = tf.cast(batch_extra, tf.float32)

        batch_delF = tf.reduce_mean(batch_pmf[:, 90:], axis=1, keepdims=True) - tf.reduce_min(batch_pmf, axis=1, keepdims=True)

        with tf.GradientTape() as tape:
            extra_pred, deltaF_pred = backext_model(batch_extra, training=True)
            backward_pred = backward_model(extra_pred, training=True)
            forward_pred = extend_model(batch_extra, training=True)

            be_pmf_loss, be_delf_loss = backext_loss((extra_pred, deltaF_pred), (batch_pmf, batch_delF))
            backward_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)(batch_sequence[:, :400], backward_pred[:, :400])
            recon_loss = pmf_reconstruction_loss(batch_pmf, forward_pred)
            total_loss = k1 * be_pmf_loss + k2 * be_delf_loss + k3 * backward_loss + k4 * recon_loss

        gradients = tape.gradient(total_loss, extend_model.trainable_variables)
        extend_model.optimizer.apply_gradients(zip(gradients, extend_model.trainable_variables))
        return total_loss, be_pmf_loss, be_delf_loss, backward_loss, recon_loss

    for epoch in range(epochs):
        callbacks.on_epoch_begin(epoch)
        epoch_components = np.zeros(4, dtype=np.float64)
        epoch_total = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, train_size)
            batch_sequence = sequence_train_data[start:end]
            batch_pmf = pmf_train_data[start:end]
            batch_extra = extra_train_data[start:end]

            total, be_pmf, be_delf, bw_loss, recon = process_batch(batch_sequence, batch_pmf, batch_extra)
            batch_len = batch_sequence.shape[0]
            epoch_total += float(total) * batch_len
            epoch_components += np.array([float(be_pmf), float(be_delf), float(bw_loss), float(recon)]) * batch_len

        epoch_total /= train_size
        epoch_components /= train_size

        delF_valid = tf.reduce_mean(pmf_valid_data[:, 90:], axis=1, keepdims=True) - tf.reduce_min(pmf_valid_data, axis=1, keepdims=True)
        extra_valid_pred, deltaF_pred = backext_model(extra_valid_data, training=False)
        backward_valid_pred = backward_model(extra_valid_pred, training=False)
        forward_valid_pred = extend_model(extra_valid_data, training=False)

        val_be_pmf, val_be_delf = backext_loss((extra_valid_pred, deltaF_pred), (pmf_valid_data, delF_valid))
        val_backward = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)(sequence_valid_data[:, :400], backward_valid_pred[:, :400])
        val_recon = pmf_reconstruction_loss(pmf_valid_data, forward_valid_pred)

        val_be_pmf = float(val_be_pmf)
        val_be_delf = float(val_be_delf)
        val_backward = float(val_backward)
        val_recon = float(val_recon)
        val_total = k1 * val_be_pmf + k2 * val_be_delf + k3 * val_backward + k4 * val_recon

        history.update(
            [epoch_total, epoch_components[0], epoch_components[2], epoch_components[3]],
            [val_total, val_be_pmf, val_backward, val_recon],
        )

        logs = {
            "trn_loss": epoch_total,
            "trn_pmf_backext_mae": epoch_components[0],
            "trn_surface_bce": epoch_components[2],
            "trn_pmf_recon_mae": epoch_components[3],
            "val_loss": val_total,
            "val_pmf_backext_mae": val_be_pmf,
            "val_surface_bce": val_backward,
            "val_pmf_recon_mae": val_recon,
        }
        callbacks.on_epoch_end(epoch, logs=logs)

    callbacks.on_train_end()
    return history
