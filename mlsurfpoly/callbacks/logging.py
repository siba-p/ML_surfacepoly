from __future__ import annotations

from timeit import default_timer as timer
from typing import Iterable, List, Sequence

import tensorflow as tf


class LogsCallback(tf.keras.callbacks.Callback):

    def __init__(self, skip_epochs: int = 10, monitor: Sequence[str] | None = None):
        super().__init__()
        self.skip_epochs = max(1, skip_epochs)
        self.monitor = list(monitor) if monitor is not None else []
        self._epoch = 0
        self._start_time = 0.0

    def on_train_begin(self, logs=None):
        self._start_time = timer()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.skip_epochs:
            return

        now = timer()
        elapsed = now - self._start_time
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)
        parts = [f"Epoch: {epoch}"]
        metrics = self.monitor or logs.keys()
        for metric in metrics:
            if metric in logs:
                parts.append(f"{metric}: {logs[metric]:.4f}")
        parts.append(f"Time for {self.skip_epochs} epochs: {minutes}min {seconds}sec")
        print("\t".join(parts))
        self._start_time = now


class LogsCallbackflex(tf.keras.callbacks.Callback):

    def __init__(self, skip_epochs: int = 10, log_items: Iterable[str] | None = None):
        super().__init__()
        self.skip_epochs = max(1, skip_epochs)
        self.log_items = list(log_items) if log_items is not None else ["loss", "val_loss"]
        self._start_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.skip_epochs == 0:
            self._start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.skip_epochs:
            return

        elapsed = timer() - self._start_time
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)
        parts = [f"\033[1;94mEpoch: {epoch}\033[0m"]
        for item in self.log_items:
            value = logs.get(item)
            if value is not None:
                parts.append(f"{item}: \033[92m{value:.4f}\033[0m")
        parts.append(f"time: \033[93m{minutes}min {seconds}sec\033[0m")
        print(", \t".join(parts))


class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, loss_metric: str, threshold: float):
        super().__init__()
        self.loss_metric = loss_metric
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get(self.loss_metric) is not None and logs[self.loss_metric] <= self.threshold:
            self.model.stop_training = True


class AnnealedSmoothBinary(tf.keras.layers.Layer):

    def __init__(self, initial_m: float = 5.0, max_m: float = 50.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_m = initial_m
        self.max_m = max_m
        self.m = tf.Variable(initial_value=initial_m, trainable=False, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        return 0.5 * (1 + tf.tanh(self.m * (inputs - 0.5)))

    def update_m(self, epoch: int):
        new_m = min(self.initial_m + epoch / 100.0, self.max_m)
        self.m.assign(tf.constant(new_m, dtype=tf.float32))

    def get_config(self):  # pragma: no cover - boilerplate
        config = super().get_config()
        config.update({"initial_m": self.initial_m, "max_m": self.max_m})
        return config


class UpdateSharpnessCallback(tf.keras.callbacks.Callback):

    def __init__(self, layer_class: type[tf.keras.layers.Layer], verbose: int = 0):
        super().__init__()
        self.layer_class = layer_class
        self.verbose = verbose
        self._layers: List[tf.keras.layers.Layer] = []

    def on_train_begin(self, logs=None):
        self._layers = self._find_layers(self.model)
        if self.verbose:
            print(f"Found {len(self._layers)} {self.layer_class.__name__} layers")

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self._layers:
            before = layer.m.numpy()
            layer.update_m(epoch + 1)
            if self.verbose > 1:
                after = layer.m.numpy()
                print(f"{layer.name}: m {before:.3f} -> {after:.3f}")

    def _find_layers(self, model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        found: List[tf.keras.layers.Layer] = []
        for layer in model.layers:
            if isinstance(layer, self.layer_class):
                found.append(layer)
            elif isinstance(layer, tf.keras.Model):
                found.extend(self._find_layers(layer))
        return found
