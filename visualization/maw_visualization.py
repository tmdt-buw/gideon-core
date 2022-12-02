import pickle
from datetime import datetime
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from guidance.recurring_pattern import Pattern
from model.mouse_event_record import MouseEventType
from model.state import State, Severity, Action


def myplot(x, y, bins):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    return heatmap.T


def bounding_box(matrix, x_scale, y_scale):
    rows = np.any(matrix, axis=1)
    cols = np.any(matrix, axis=0)
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min = x_min * x_scale
    x_max = x_max * x_scale
    y_min = y_min * y_scale
    y_max = y_max * y_scale
    return (x_min, y_min), x_max - x_min, y_max - y_min


def pattern_class_to_string(pattern_class: int):
    if pattern_class == 0:
        return 'Overview of label classes'
    elif pattern_class == 1:
        return 'Variants of normal curves'
    elif pattern_class == 2:
        return 'Variants of warning curves'
    elif pattern_class == 3:
        return 'Variants of error curves'


def colorForSeverity(severity: Severity):
    if severity == Severity.okay:
        return 'tab:green'
    if severity == Severity.warning:
        return 'tab:orange'
    return 'tab:red'


def visualize_state_list(states: List[State], actions: List[Action], pattern_class: int, i):
    num_rows = len(states)
    num_cols = max([len(state.charts) for state in states])

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 4), dpi=400)
    fig.suptitle(pattern_class_to_string(pattern_class), fontsize=24)
    axes_to_remove = []
    legend_lines = []
    legend_names = []
    legend_names.append('voltage')
    legend_lines.append(Line2D([0], [0], color='tab:cyan', lw=4))
    legend_names.append('current')
    legend_lines.append(Line2D([0], [0], color='tab:olive', lw=4))
    for row in range(num_rows):
        state = states[row]
        num_charts = len(state.charts)
        for column in range(num_cols):
            if column < num_charts:
                chart = state.charts[column]
                if len(chart.data) == 0:
                    continue
                # x = np.arange(0, len(chart.data[0]), 1)
                if num_cols == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, column]
                if column == 0:
                    ax.set_ylabel(f'State: {row + 1}\nAction: {actions[row].type} {actions[row].parameters}')
                ax2 = ax.twinx()
                ax2.get_xaxis().set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, p: f"{datetime.fromtimestamp(x).second}.{int(datetime.fromtimestamp(x).microsecond / 1000)}")
                )
                y_dim = 60
                x_values = []
                for idx, chart_data in enumerate(chart.data):
                    data = list(zip(*chart_data))
                    x_values = [float(date) for date in data[0]]
                    y_values = list(data[1])
                    if idx == 0:
                        ax.plot(x_values, y_values, 'tab:cyan', linewidth=1)
                    else:
                        ax2.plot(x_values, y_values, 'tab:olive', linewidth=1)
                for label in chart.labels:
                    labels_xs = []
                    for idx, date in enumerate(x_values):
                        if label.start <= date <= label.end:
                            labels_xs.append(date)
                    if not labels_xs:
                        continue
                    ax.fill_between(labels_xs, 0, 1, color=colorForSeverity(label.label_class.severity), alpha=0.2, transform=ax.get_xaxis_transform())
                    if label.label_class.name not in legend_names:
                        legend_names.append(label.label_class.name)
                        legend_lines.append(Line2D([0], [0], color=colorForSeverity(label.label_class.severity), alpha=0.5, lw=4))
                xs = [x_values[0], x_values[-1]]
                ys = [0.0, 60.0]
                for event in chart.events:
                    if event.type is MouseEventType.mousemove or event.type is MouseEventType.mouseover:
                        xs.append(event.x * (x_values[-1] - x_values[0]) + x_values[0])
                        ys.append(y_dim * event.y)
                bins = 50
                H = myplot(xs, ys, bins)
                H_masked = np.ma.masked_array(H, H < 3)
                if H_masked.any():
                    x_scale = (x_values[-1] - x_values[0]) / bins
                    y_scale = 60.0 / bins
                    origin, width, height = bounding_box(H_masked, x_scale, y_scale)
                    adj_origin = (origin[0] + x_values[0], origin[1])
                    # Create a Rectangle patch
                    rect = Rectangle(adj_origin, width, height, linewidth=1, edgecolor='b', facecolor='none', zorder=2)
                    # Add the patch to the Axes
                    ax.add_patch(rect)
            else:
                axes_to_remove.append(axes[row, column])

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    if num_cols == 1:
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape)
    else:
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="lightgray")
        fig.add_artist(line)

    for axe in axes_to_remove:
        axe.remove()

    plt.figlegend(legend_lines, legend_names, loc=(0.8, 0.8))
    plt.savefig(f'fig{i}.png')
    # plt.show()


def visualize_eda_session(adapted_patterns: List[List[State]], actions_list: List[List[Action]], patterns: List[Pattern]):
    i = 7
    for adapted_pattern, actions, pattern in zip(adapted_patterns, actions_list, patterns):
        visualize_state_list(adapted_pattern, actions, pattern.pattern_class, i)
        i += 1


if __name__ == '__main__':
    with open("../data/metalarcwelding/interaction/adapted_patterns.pkl", "rb") as f:
        adapted_patterns: List[List[State]] = pickle.load(f)

    with open("../data/metalarcwelding/interaction/classified_patterns.pkl", "rb") as f:
        patterns: List[Pattern] = pickle.load(f)

    visualize_eda_session(adapted_patterns, actions_list, patterns)
