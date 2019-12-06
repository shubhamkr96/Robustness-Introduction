import tensorflow as tf
from tensorflow import keras


def evaluate_adversarial(model, test_dataset, epsilons):
    evaluation = []
    for epsilon in epsilons:
        metric = keras.metrics.Accuracy()
        misclassified_logits = []
        top_misclassified_images = []

        for i, (images, targets) in enumerate(test_dataset):
            w, b = model.trainable_variables
            w = tf.reshape(w, [1, -1])  # Adjust correctly dimensions
            targets = tf.reshape(targets, [-1, 1])
            delta = targets * -epsilon * tf.sign(w)
            adversarial_images = images + tf.reshape(delta, images.shape)

            prediction = model(adversarial_images)
            predicted_labels = tf.sign(prediction)
            metric(targets, predicted_labels)

            # Update top k misclassified images (closest to the boundary)
            top_misclassified_images = collect_misclassified_images(
                top_misclassified_images, images, targets, prediction, predicted_labels, delta, k=10
            )
            misclassified_logits.extend(
                logits.numpy()[0] for y, logits, y_hat in zip(targets, prediction, predicted_labels)
                if y != y_hat)

        accuracy = metric.result().numpy()
        evaluation.append([epsilon, accuracy, top_misclassified_images, misclassified_logits])
    return evaluation


def collect_misclassified_images(misclassified, images, targets, prediction, predicted_labels, delta, k=10):
    misclassified.extend([(image.numpy(), logits.numpy()[0], adversarial_image.numpy())
                          for image, y, logits, y_hat, adversarial_image
                          in
                          zip(images, targets, prediction, predicted_labels, images + tf.reshape(delta, images.shape))
                          if y != y_hat])
    misclassified = sorted(misclassified, key=lambda tup: abs(tup[1]))
    misclassified = misclassified[:k]  # Leave only first top 10 candidates (with the smallest lost)
    return misclassified

