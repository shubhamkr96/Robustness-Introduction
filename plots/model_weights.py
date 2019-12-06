from matplotlib import pyplot as plt
from tensorflow import keras
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Set up data
model_regular = keras.models.load_model('../out/regular/model.h5')
model_robust_1 = keras.models.load_model('../out/robust/model.0.1.h5')
model_robust_2 = keras.models.load_model('../out/robust/model.0.2.h5')
model_robust_3 = keras.models.load_model('../out/robust/model.0.3.h5')
models = [model_regular, model_robust_1, model_robust_2, model_robust_3]
labels = ['Regular', r'Robust ($\epsilon=0.1$)', r'Robust ($\epsilon=0.2$)', r'Robust ($\epsilon=0.3$)']

no_ticks = {'xticks': [], 'yticks': []}
fig, axes = plt.subplots(1, 4, figsize=[6, 2], subplot_kw=no_ticks)

for ax, model, label in zip(axes, models, labels):
    w, b = model.trainable_variables
    ax.imshow(1 - w.numpy().reshape(28, 28), cmap='gray')
    ax.set_title(label, fontsize=7)

plt.suptitle('Model Weights (W)')
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Make space for the title
name, ext = __file__.split('.')
plt.savefig(f'{name}.png', dpi=300)
