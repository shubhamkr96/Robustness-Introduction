import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from source import callbacks, routines, utils, data, evaluate
tf.random.set_seed(seed=1)


def robust_hinge_loss(model, y_hat, y):     # Inject the epsilon
    w, b = model.trainable_variables
    hinge = lambda x: tf.maximum(0, 1-x)    # The raw hinge function
    delta = -epsilon * tf.norm(w, ord=1)    # Use the inf ball
    z1 = y * tf.reshape(y_hat, [-1])
    z2 = tf.add(z1, delta)
    loss_value = hinge(z2)
    return loss_value


def train_step(inputs, targets):            # Inject dependencies: a model, and an optimizer
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        loss_value = robust_hinge_loss(model, prediction, targets)      # Change the loss function
    gradients = tape.gradient(loss_value, model.trainable_variables)
    sgd_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    predicted_labels = tf.sign(prediction)
    return predicted_labels, loss_value, gradients


def test_step(inputs, targets):
    prediction = model(inputs)
    loss_value = hinge_loss(prediction, targets)
    predicted_labels = tf.sign(prediction)
    return predicted_labels, loss_value


# Run the experiment
if __name__ == '__main__':
    # Set up directory
    root_dir = os.path.abspath(os.path.dirname(__file__))
    experiment_dir = os.path.join(root_dir, 'out', 'robust')
    os.chdir(experiment_dir)

    # Convert Multi-class problem to binary
    train_dataset = data.binary_mnist(split='train').batch(100)  # In fact, we should do three folded split
    test_dataset = data.binary_mnist(split='test').batch(1000)  # to make a valid test.

    # The logger is the same for each epsilon (sub-experiments)
    cb_loss_logger = callbacks.LossLogger('experiment.log')
    logger = logging.getLogger('callbacks')

    for epsilon in [0.1, 0.2, 0.3]:
        logger.info(f'Train the robust model, epsilon: {epsilon}')

        # Build and compile linear model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(1),
        ])
        sgd_optimizer = keras.optimizers.SGD(learning_rate=.0001)   # Norm w added to the loss, avoid gradient exploding
        hinge_loss = keras.losses.Hinge()  # This is: max(0, 1 - y_true * y_pred), where y_true in {+1, -1}

        # Build callbacks to track process
        cb_accuracy = callbacks.Accuracy()
        cb_gradients = callbacks.CollectGradients()

        routines.train(train_step, train_dataset, test_step, test_dataset,
                       epochs=10, callbacks=(cb_accuracy, cb_loss_logger, cb_gradients))
        evaluation = evaluate.evaluate_adversarial(model, test_dataset, epsilons=np.linspace(0, 0.5, num=101))

        # Save the experiment
        model.save(f'model.{epsilon}.h5')
        utils.save([evaluation, cb_accuracy, cb_gradients], f'results.{epsilon}.bin')
