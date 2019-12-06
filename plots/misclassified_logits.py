import numpy as np
import statsmodels.nonparametric.api as smnp
import matplotlib.pyplot as plt
from source import utils
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def prepare(name):
    evaluation_robust, *_ = utils.load(name)
    data = [(epsilon, misclassified_logits) for epsilon, *_, misclassified_logits in evaluation_robust[5::5]]
    return data


def density(data):
    x = np.array(data, dtype=np.float64)
    kde = smnp.KDEUnivariate(x)
    kde.fit("gau", bw=.5, fft=True)
    x, y = kde.support, kde.density
    return x, y


def plot():
    global origin_elevation

    for i, (epsilon, x) in enumerate(data):
        x, y = density(x)
        y -= i * space
        ax.plot(x, y, color='black', alpha=0.7)
        ax.fill_between(x, -i * space, y, facecolor='black', alpha=.1)

        if not origin_elevation:
            origin_elevation = min(x) - ax_space

        if any(abs(i-epsilon) < .01 for i in np.linspace(0, .5, 11)):   # Conversion type problems
            ax.text(min(x) - text_space, -i * space, f'{epsilon:.2f}', size=8, ha="right", va="bottom")

    ax.text(label_x, -i * space / 2, r'Perturbations ($\epsilon$)',
            ha="center", va="center", rotation=label_rotation)
    ax.arrow(origin_elevation, 0, min(x) - origin_elevation - ax_space, -i * space,
             head_width=head_width, head_length=head_length, linewidth=.1, color='black', length_includes_head=True)
    ax.set_title(title)

    # Adjust the view
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - x_lim, xmax)

    # Reduce number of ticks
    ax.set_yticks([])

    # Eliminate upper, left and right axes
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_linewidth(0.3)


fig, axes = plt.subplots(1, 2, figsize=[10, 4])

# Regular Model
origin_elevation = None
space = 0.1
ax_space = 4.5
text_space = 5.5
label_rotation = 52
label_x = -15
x_lim = 4
head_width = .02
head_length = .5
ax = axes[0]
title = 'Regular Model'
data = prepare('../out/regular/results.bin')
plot()


# Robust Model, epsilon 0.2
origin_elevation = None
space = 0.1
ax_space = 1.7
text_space = 2.1
label_rotation = 61
label_x = -5.4
x_lim = 2
head_width = .02
head_length = .15
ax = axes[1]
title = r'Robust Model ($\epsilon=0.2$)'
data = prepare('../out/robust/results.0.2.bin')
plot()


plt.suptitle(f'Misclassified Logits Distribution')

plt.tight_layout(rect=[0, 0, 1, 0.9])  # Make space for the title
name, ext = __file__.split('.')
plt.savefig(f'{name}.png', dpi=300)
