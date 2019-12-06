import numpy as np
import plotly.graph_objects as go
np.random.seed(1)

x = np.random.normal(3, .7, 50)
y = np.random.normal(3, .7, 50)
z = np.random.normal(3, .7, 50)


def get_box(xi, yi, zi):
    return go.Mesh3d(
        # 8 vertices of a cube
        x=[xi-e, xi-e, xi+e, xi+e, xi-e, xi-e, xi+e, xi+e],
        y=[yi-e, yi+e, yi+e, yi-e, yi-e, yi+e, yi+e, yi-e],
        z=[zi-e, zi-e, zi-e, zi-e, zi+e, zi+e, zi+e, zi+e],
        color='rgb(120, 184, 120)',
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.3,
    )


def get_box_boarder(xi, yi, zi):
    return go.Scatter3d(
        x=[xi-e, xi+e, xi+e, xi-e, xi-e, xi-e, xi+e, xi+e, xi-e, xi-e, xi+e, xi+e, xi+e, xi+e, xi-e, xi-e],
        y=[yi-e, yi-e, yi+e, yi+e, yi-e, yi-e, yi-e, yi+e, yi+e, yi-e, yi-e, yi-e, yi+e, yi+e, yi+e, yi+e],
        z=[zi-e, zi-e, zi-e, zi-e, zi-e, zi+e, zi+e, zi+e, zi+e, zi+e, zi+e, zi-e, zi-e, zi+e, zi+e, zi-e],
        mode='lines',
        line=dict(color='rgb(50, 50, 50)', width=2)
    )


e = 0.2
boxes = [get_box(xi, yi, zi) for xi, yi, zi in zip(x, y, z)]
boarders = [get_box_boarder(xi, yi, zi) for xi, yi, zi in zip(x, y, z)]
scatter = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(50, 50, 50)',
    )
)

fig = go.Figure(data=[scatter, *boxes, *boarders])
fig.update_layout(
    scene=dict(
        xaxis_title="",
        yaxis_title="",
        zaxis_title=""
    ),
    template='simple_white',
    scene_xaxis_showticklabels=False,
    scene_yaxis_showticklabels=False,
    scene_zaxis_showticklabels=False,
)
fig.show()
