from matplotlib import pyplot as plt
from source import utils
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Set up data
evaluation_regular, *_ = utils.load('../out/regular/results.bin')
evaluation_robust_1, *_ = utils.load('../out/robust/results.0.1.bin')
evaluation_robust_2, *_ = utils.load('../out/robust/results.0.2.bin')
evaluation_robust_3, *_ = utils.load('../out/robust/results.0.3.bin')


def prepare(evaluation):
    accuracies = [(epsilon, accuracy) for epsilon, accuracy, *_ in evaluation]
    return zip(*accuracies)


fig, ax = plt.subplots(figsize=[7, 5])

x, y = prepare(evaluation_regular)
ax.plot(x, y, color='black', linewidth=0.5)
ax.plot(x, y, color='black', alpha=0.3, linewidth=2)
ax.fill_between(x, 0, y, facecolor='black', alpha=.05, label='Regular')

old_y = y
x, y = prepare(evaluation_robust_1)
ax.plot(x, y, color='black', linewidth=0.5)
ax.plot(x, y, color='green', alpha=0.3, linewidth=2)
ax.fill_between(x, old_y, y, facecolor='green', alpha=.1, label=r'Robust ($\epsilon=0.1$)')

old_y = y
x, y = prepare(evaluation_robust_2)
ax.plot(x, y, color='black', linewidth=0.5)
ax.plot(x, y, color='green', alpha=0.5, linewidth=2)
ax.fill_between(x, old_y, y, facecolor='green', alpha=.2, label=r'Robust ($\epsilon=0.2$)')

old_y = y
x, y = prepare(evaluation_robust_3)
ax.plot(x, y, color='black', linewidth=0.5)
ax.plot(x, y, color='green', alpha=0.7, linewidth=2)
ax.fill_between(x, old_y, y, facecolor='green', alpha=.3, label=r'Robust ($\epsilon=0.3$)')

ax.set_xlabel(r'Perturbation Scale ($\epsilon$)')
ax.set_ylabel('Accuracy')
ax.set_title('White-Box Adversarial Attacks')

plt.legend()
name, ext = __file__.split('.')
plt.savefig(f'{name}.png', dpi=300)
