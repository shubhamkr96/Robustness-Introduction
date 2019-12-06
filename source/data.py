import tensorflow as tf
import tensorflow_datasets as tfds


# Load MNIST Dataset
mnist_builder = tfds.builder('mnist')
mnist_builder.download_and_prepare()  # more details: mnist_builder.info


def binary_mnist(split: str) -> tf.data.Dataset:
    mnist = mnist_builder.as_dataset(split=split)
    return mnist.filter(filter_binary_mnist).map(change_mnist_labels)


def filter_binary_mnist(sample):
    image, label = sample['image'], sample['label']
    return tf.logical_or(tf.equal(label, 0), tf.equal(label, 1))


def change_mnist_labels(sample):
    cast = lambda x: tf.cast(x, tf.float32)
    image, label = cast(sample['image']) / 255, cast(sample['label'])
    if tf.equal(label, cast(0)):
        label = cast(-1)
    return image, label
