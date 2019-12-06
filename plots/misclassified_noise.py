import numpy as np
from matplotlib import pyplot as plt
from source import utils
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Set up data
evaluation_regular, *_ = utils.load('../out/regular/results.bin')
evaluation_robust_1, *_ = utils.load('../out/robust/results.0.1.bin')
evaluation_robust_2, *_ = utils.load('../out/robust/results.0.2.bin')
evaluation_robust_3, *_ = utils.load('../out/robust/results.0.3.bin')
evaluations = [evaluation_regular, evaluation_robust_1, evaluation_robust_2, evaluation_robust_3]


no_ticks = {'xticks': [], 'yticks': []}
fig, axes = plt.subplots(len(evaluations), 11, figsize=[9, 3], subplot_kw=no_ticks)

# Plot misclassified samples from the regular model
for ax_row, evaluation in zip(axes, evaluations):

    evaluation = evaluation[::10]
    for i, ax in enumerate(ax_row):
        epsilon, accuracy, top_misclassified_images, misclassified_logits = evaluation[i]
        image, logit, adversarial_image = top_misclassified_images[0]
        ax.imshow(1 - adversarial_image[..., 0], cmap='gray')

axes[0, 0].text(-50, 14, 'Regular Model', size=8, ha="center", va="center")
axes[1, 0].text(-50, 14, r'Robust Model ($\epsilon=0.1$)', size=8, ha="center", va="center")
axes[2, 0].text(-50, 14, r'Robust Model ($\epsilon=0.2$)', size=8, ha="center", va="center")
axes[3, 0].text(-50, 14, r'Robust Model ($\epsilon=0.3$)', size=8, ha="center", va="center")

for i, epsilon in enumerate(np.linspace(0, 0.50, 11)):
    axes[3, i].text(14, 40, r'$\epsilon=$' f'{epsilon:.2f}', size=8, ha="center", va="center")

plt.tight_layout()
name, ext = __file__.split('.')
plt.savefig(f'{name}.png', dpi=300)
