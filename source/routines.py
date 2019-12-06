from typing import Iterable, Callable, Tuple
from . import callbacks as cb


def train(train_step: Callable,
          train_dataset: Iterable,
          test_step: Callable,
          test_dataset: Iterable,
          epochs: int = 10,
          callbacks: Tuple = ()):
    callbacks = cb.CallbackList(callbacks)
    for epoch in range(epochs):
        callbacks.on_epoch_begin(epoch)
        train_loop(train_step, train_dataset, callbacks)
        test_loop(test_step, test_dataset, callbacks)
        callbacks.on_epoch_end(epoch)


def train_loop(train_step: Callable, dataset: Iterable, callbacks: cb.Callback):
    for i, (inputs, targets) in enumerate(dataset):
        prediction, loss_value, gradients = train_step(inputs, targets)
        callbacks.on_train_batch_end(i, inputs, targets, prediction, loss_value, gradients)
    callbacks.on_train_end()


def test_loop(test_step: Callable, dataset: Iterable, callbacks: cb.Callback):
    for i, (inputs, targets) in enumerate(dataset):
        prediction, loss_value = test_step(inputs, targets)
        callbacks.on_test_batch_end(i, inputs, targets, prediction, loss_value)
    callbacks.on_test_end()
