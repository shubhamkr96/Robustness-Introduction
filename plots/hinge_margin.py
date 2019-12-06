import numpy as np
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Collect data
x = np.linspace(-3, 3, num=1000)
y0 = (1+x).clip(min=0)
y1 = (1-x).clip(min=0)

# Build figure
fig, ax = plt.subplots(figsize=[4, 3])

# Plot data
ax.plot(x, y0, color='black', linestyle='--', linewidth=2)
ax.plot(x, y1, color='black', linewidth=2)

# Plot soft-margin
ymin, ymax = ax.get_ylim()
ax.axvspan(-1, 1, alpha=.1, facecolor='green')
ax.axvline(1, ymin, ymax, linestyle='--', linewidth=.3, dashes=[10, 4], color='black')
ax.axvline(-1, ymin, ymax, linestyle='--', linewidth=.3, dashes=[10, 4], color='black')
ax.text(0, 3, r'Soft-margin', size=8, ha="center", va="center",
        bbox=dict(facecolor='white'))

# Reduce number of ticks
ax.set_xticks([1])
ax.set_yticks([1])

# Set labels
ax.set_xlabel(r'$\hat{y}$')
ax.set_ylabel(r'L', rotation=0)

# Add texts
ax.text(-3.2, 4.1, r'$y=1$', size=8)
ax.text(2.7, 4.1, r'$y=-1$', size=8)

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Thicker axes
ax.spines['left'].set_linewidth(0.3)
ax.spines['bottom'].set_linewidth(0.3)

# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')

# Adjust positions of ax labels
ax.xaxis.set_label_coords(1.02, -.02)
ax.yaxis.set_label_coords(.52, 1.02)

# Add head arrows to axes
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.arrow(0, 0, 0, ymax-.1, head_width=.05, head_length=.1, linewidth=.1, color='black')
ax.arrow(0, 0, xmax-.1, 0, head_width=.05, head_length=.1, linewidth=.1, color='black')

# Make equal aspect
ax.set_aspect('equal')

plt.tight_layout()
name, ext = __file__.split('.')
plt.savefig(f'{name}.png', dpi=300)
