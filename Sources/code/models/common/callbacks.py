import math
import tensorflow as tf

class ParameterScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 param_name: str,
                 start_value: float,
                 end_value: float,
                 schedule_epochs: int,
                 schedule: str = 'linear',
                 verbose: bool = True):

        super().__init__()
        self.param_name = param_name
        self.start = start_value
        self.end = end_value
        self.epochs = schedule_epochs
        self.schedule = schedule
        self.verbose = verbose

    def _compute(self, epoch: int) -> float:
        fraction = min(epoch / float(self.epochs), 1.0)
        if self.schedule == 'linear':
            return self.start + (self.end - self.start) * fraction

        elif self.schedule == 'sigmoid':
            x = (epoch - self.epochs / 2) / (self.epochs / 10)
            return self.end / (1 + math.exp(-x))

        elif self.schedule == 'cosine':
            cos_out = (1 + math.cos(math.pi * fraction)) / 2
            return self.end + (self.start - self.end) * cos_out

        else:
            return self.end

    def on_epoch_begin(self, epoch, logs=None):
        new_val = float(self._compute(epoch))
        setattr(self.model, self.param_name, new_val)
        if self.verbose:
            print(f"\nEpoch {epoch+1}: set {self.param_name} = {new_val:.6f}")