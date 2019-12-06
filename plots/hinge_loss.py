import numpy as np
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Collect data
x = np.linspace(-3, 3, num=1000)
y = (1-x).clip(min=0)

# Build figure
fig, ax = plt.subplots(figsize=[4, 3])

# Plot data
ax.plot(x, y, color='black', linewidth=2)
ax.fill_between(x, 0, y, facecolor='black', alpha=.05)

# Reduce number of ticks
ax.set_xticks([1])
ax.set_yticks([1])

# Set labels
ax.set_xlabel(r'$y \cdot \hat{y}$')
ax.set_ylabel(r'L', rotation=0)

# Add text
ax.text(-1.7, 1, r'$max(0, 1-y \cdot \hat{y})$', size=8, ha="center", va="center",
        bbox=dict(facecolor='white'))

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Thicker axes
# ax.spines['left'].set_linewidth(0.1)
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
