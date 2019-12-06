import logging
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from tensorflow import keras
logger = logging.getLogger('callbacks')


class Callback:

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_batch_end(self, batch, inputs, targets, prediction, loss_value, gradients):
        pass

    def on_train_end(self):
        pass

    def on_test_batch_end(self, batch, inputs, targets, prediction, loss_value):
        pass

    def on_test_end(self):
        pass


@dataclass
class CallbackList(Callback):
    callbacks: Tuple[Callback]

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_batch_end(self, *args):
        for callback in self.callbacks:
            callback.on_train_batch_end(*args)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_test_batch_end(self, *args):
        for callback in self.callbacks:
            callback.on_test_batch_end(*args)

    def on_test_end(self):
        for callback in self.callbacks:
            callback.on_test_end()


@dataclass
class LossLogger(Callback):
    _epoch: int = 0
    _train_loss: keras.metrics.Mean = None
    _test_loss: keras.metrics.Mean = None
    level: int = 20

    def __init__(self, file_path: str = None, verbose=0):
        self.verbose = verbose
        logger.setLevel(self.level)
        logger.propagate = False
        formatter = logging.Formatter('%(asctime)s [%(levelname)-6s] [%(name)-10s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)      # handle all messages from logger (not set handler level)
        if file_path:
            file_handler = logging.FileHandler(file_path, mode='w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def on_epoch_begin(self, epoch):
        self._epoch = epoch
        self._train_loss = keras.metrics.Mean()
        self._test_loss = keras.metrics.Mean()

    def on_train_batch_end(self, batch, inputs, targets, prediction, loss_value, gradients):
        self._train_loss(loss_value)
        batch_loss = np.mean(loss_value)
        if self.verbose == 1:
            message = f'Epoch {self._epoch:3d}     Batch{batch:4d}     Train Loss: {batch_loss:.5f}'
            logger.info(message)

    def on_train_end(self):
        loss = self._train_loss.result()
        message = f'Epoch {self._epoch:3d}    Average Train Loss: {loss:.5f}'
        logger.info(message)
        self._train_loss = None

    def on_test_batch_end(self, batch, inputs, targets, prediction, loss_value):
        self._test_loss(loss_value)

    def on_test_end(self):
        loss = self._test_loss.result()
        message = f'Epoch {self._epoch:3d}    Average Test  Loss: {loss:.5f}'
        logger.info(message)
        self._test_loss = None


@dataclass
class Accuracy(Callback):
    _epoch: int = 0
    _train_accuracy: keras.metrics.Accuracy = None
    _test_accuracy: keras.metrics.Accuracy = None
    train: dict = field(default_factory=dict)
    train_details: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)

    def on_epoch_begin(self, epoch):
        self._epoch = epoch
        self._train_accuracy = keras.metrics.Accuracy()
        self._test_accuracy = keras.metrics.Accuracy()
        self.train_details[epoch] = []

    def on_train_batch_end(self, batch, inputs, targets, prediction, loss_value, gradients):
        self._train_accuracy(targets, prediction)
        new_metric = keras.metrics.Accuracy()
        new_metric(targets, prediction)
        batch_accuracy = new_metric.result().numpy()
        self.train_details[self._epoch].append(batch_accuracy)

    def on_test_batch_end(self, batch, inputs, targets, prediction, loss_value):
        self._test_accuracy(targets, prediction)

    def on_epoch_end(self, epoch):
        self.train[epoch] = self._train_accuracy.result()
        self.test[epoch] = self._test_accuracy.result()
        message = f'Epoch {self._epoch:3d}    Average Train error: {1-self.train[epoch]:.5f}'
        logger.info(message)
        message = f'Epoch {self._epoch:3d}    Average Test  error: {1-self.test[epoch]:.5f}'
        logger.info(message)
        self._train_accuracy = self._test_accuracy = None   # Enable pickle


@dataclass
class CollectGradients(Callback):
    _epoch: int = 0
    data: dict = field(default_factory=dict)

    def on_epoch_begin(self, epoch):
        self._epoch = epoch
        self.data[epoch] = []

    def on_train_batch_end(self, batch, inputs, targets, prediction, loss_value, gradients):
        self.data[self._epoch].append([grad.numpy() for grad in gradients])
