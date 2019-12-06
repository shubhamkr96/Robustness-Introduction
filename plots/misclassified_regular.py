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
fig, axes = plt.subplots(4, 4, figsize=[3, 3], subplot_kw=no_ticks)

# Plot misclassified samples from the regular model
for ax_row, evaluation in zip(axes, evaluations):
    epsilon, accuracy, top_misclassified_images, misclassified_logits = evaluation[0]
    for i, ax in enumerate(ax_row):
        try:
            image, logit, adversarial_image = top_misclassified_images[i]
            ax.imshow(1 - adversarial_image[..., 0], cmap='gray')
        except IndexError:
            ax.axis('off')

axes[0, 0].text(-10, 14, 'Regular', size=8, ha="right", va="center")
axes[1, 0].text(-10, 14, r'$\epsilon=0.1$', size=8, ha="right", va="center")
axes[2, 0].text(-10, 14, r'$\epsilon=0.2$', size=8, ha="right", va="center")
axes[3, 0].text(-10, 14, r'$\epsilon=0.3$', size=8, ha="right", va="center")

plt.suptitle(f'Misclassified Examples')
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Make space for the title
name, ext = __file__.split('.')
plt.savefig(f'{name}.png', dpi=300)
