import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from star_scale.trainingfuncs import *
from star_scale.math_utils import *
from star_scale.utils import *

def plot_psychometric(data,
                      x,
                      y,
                      hue,
                      scatter_kws={"s": 1},
                      fit_reg='4pl',
                      x_bins=7,
                      height=3,
                      aspect=1,
                      row="lesser_dim",
                      col="greater_dim",
                      row_order=None,
                      col_order=None,
                      legend_title="Subject",
                      x_label="Morph Position",
                      y_label="P(aligned response)",
                      divisions=4,
                      custom_palette=None,
                      legend=True,
                     **kwargs):
    
    if row and row_order is None:
        row_order = np.sort(data[row].unique())
    if col and col_order is None:
        col_order = np.sort(data[col].unique())
    if custom_palette is None:
        custom_palette = sns.color_palette("tab10")
        
    g = sns.lmplot(
        x=x,
        y=y,
        x_bins=x_bins,
        row=row,
        col=col,
        hue=hue,
        data=data,
        fit_reg=False,
        row_order=row_order,
        col_order=col_order,
        height=height,
        aspect=aspect,
        scatter_kws=scatter_kws,
        palette=custom_palette,
        sharex=False,
        legend=False,
        **kwargs
        )
    if fit_reg=='4pl':
        g.map_dataframe(fourpl, x, y)
    elif fit_reg=='4pl_norm':
        g.map_dataframe(fourpl_norm, x, y)
    elif fit_reg=='2pl':
        g.map_dataframe(twopl, x, y)
    
    if row is "lesser_dim" and col is "greater_dim":
        morph_dims = data["morph_dim"].unique()
        format_morph_dim_label(g, row_order, col_order, morph_dims)
    elif hue is "morph_dim":
        format_morph_dim_ax_label(g.axes.flatten()[0])
    elif col is "morph_dim":
        for col_index, morph_dim in enumerate(col_order):
            ax = g.axes.flatten()[col_index]
            format_morph_dim_ax_label(ax, morph_dim=morph_dim)
    else:
        print("I'll need to do something about these axis labels")
    g = g.set_titles('')
    g = g.set(xlim=(1, 128), ylim=(0, 1), yticks=[0.0, 0.5, 1.0])
    if legend:
        g = g.add_legend(title=legend_title)
    g = g.set_axis_labels(x_label, y_label)
    return g

def format_morph_dim_label(
    g, row_order, col_order, morph_dims, flip=False, x_axis=True, **kwargs
):
    for row_index in range(len(row_order)):
        for col_index in range(len(col_order)):
            morph_dim = row_order[row_index] + col_order[col_index]
            if morph_dim in morph_dims:
                if flip:
                    morph_dim = morph_dim[::-1]
                format_morph_dim_ax_label(
                    g.axes[row_index, col_index],
                    morph_dim=morph_dim,
                    x_axis=x_axis,
                    **kwargs
                )
            else:
                if x_axis:
                    g.axes[row_index, col_index].set_xticks([])
                else:
                    g.axes[row_index, col_index].set_yticks([])


def format_morph_dim_ax_label(ax, morph_dim="", x_axis=True, divisions=4):
    if x_axis:
        ax.set_xticks([1, 128])
        ax.set_xticklabels(morph_dim.upper())
        ax.set_xticks(
            np.linspace(1, 128, divisions, endpoint=False)[1:], minor=True
        )
    else:
        ax.set_yticks([1, 128])
        ax.set_yticklabels(morph_dim.upper())
        ax.set_yticks(
            np.linspace(1, 128, divisions, endpoint=False)[1:], minor=True
        )
        
def fourpl_norm(x, y, color=None,**kwargs):
    data = kwargs.pop("data")

    result = fit_4pl(data[x].values, data[y].values.astype(np.double))
    try:
        result_4pl = four_param_logistic([0.01, 0.99]+list(result[2:]))
        t = np.arange(128) + 1

        if color is None:
            lines, = plt.plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()

        plt.plot(t, result_4pl(t), color=color)
    except TypeError:
        pass