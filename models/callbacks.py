from timeit import default_timer as timer
import tensorflow as tf
class LogsCallback(tf.keras.callbacks.Callback):
    def __init__(self, skip_epochs, monitor=["val_mae", "val_mse"]):
        """
        Args:
            skip_epochs: Print stats every this many epochs
            monitor: List of metrics to monitor (e.g., ["val_mae", "val_mse"])
                     If None, will print all available metrics in logs
        """
        super().__init__()
        self.skip_epochs = skip_epochs
        self.epoch = 0
        self.start_time = 0.0
        self.time_block_duration = "0 sec"
        self.monitor = monitor if monitor is not None else []

    def on_train_begin(self, logs=None):
        self.start_time = timer()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.skip_epochs == 0:
            # Calculate time duration
            current_time = timer()
            elapsed_sec = int((current_time - self.start_time) % 60)
            elapsed_min = int((current_time - self.start_time) / 60.0)
            self.time_block_duration = f"{elapsed_min}min {elapsed_sec}sec"
            
            # Prepare output string
            output_parts = [f"Epoch: {epoch}"]
            
            # Add all monitored metrics or all available metrics if none specified
            metrics_to_show = self.monitor if self.monitor else logs.keys()
            
            for metric in metrics_to_show:
                if metric in logs:
                    output_parts.append(f"{metric}: {logs[metric]:.4f}")
            
            # Add timing information
            output_parts.append(f"Time for {self.skip_epochs} epochs: {self.time_block_duration}")
            
            # Print the formatted output
            print("\t".join(output_parts))
            
            # Reset timer for next block
            self.start_time = timer()

class LogsCallbackflex(tf.keras.callbacks.Callback):
    def __init__(self, skip_epochs=10, log_items=None):

        super().__init__()
        self.skipsteps = skip_epochs
        self.log_items = log_items if log_items is not None else ["loss", "val_loss"]
        self.epoch = 0
        self.starttime = 0.0
        self.endtime = 0.0
        self.timebd = "0 sec"

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.skipsteps == 0:
            if epoch != 0:
                self.endtime = timer()
                self.timesec = int((self.endtime - self.starttime) % 60)
                self.timemin = int((self.endtime - self.starttime) / 60.0)
                self.timebd = f"{self.timemin}min. {self.timesec}sec."
            log_str = f"\033[1;94mEpoch: {self.epoch}\033[0m"
            for item in self.log_items:
                value = logs.get(item, None)
                if value is not None:
                    log_str += f", \t{item}: \033[92m{value:.4f}\033[0m"
            log_str += f", \ttime for prev. {self.skipsteps} epochs: \033[93m{self.timebd}\033[0m"
            print(log_str)
            self.starttime = timer()

class CustomCallback(tf.keras.callbacks.Callback):
    loss_metric = ""
    val = 0.0

    def __init__(self, loss_metric, val):
        self.loss_metric = loss_metric
        self.val = val

    def on_epoch_end(self, epoch, logs=None):
        if logs.get(self.loss_metric) <= self.val:
            self.model.stop_training = True  # Early stopping callbacks

