import pickle
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from guidance.recurring_pattern import Pattern
from model.mouse_event_record import MouseEventType
from model.state import State, Severity, Action, Session


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
    for row in range(num_rows):
        state = states[row]
        num_charts = len(state.charts)
        for column in range(num_cols):
            if column < num_charts:
                chart = state.charts[column]
                if len(chart.data) == 0:
                    continue
                x = np.arange(0, len(chart.data[0]), 1)
                color = colorForSeverity(Severity.okay)
                for label in chart.labels:
                    label_name = label.label_class.name
                    severity = label.label_class.severity
                    if label_name not in legend_names:
                        legend_names.append(label_name)
                        legend_lines.append(Line2D([0], [0], color=colorForSeverity(severity), lw=4))
                    if severity == Severity.warning and colorForSeverity(Severity.okay):
                        color = colorForSeverity(severity)
                    if severity == Severity.error:
                        color = colorForSeverity(severity)
                        break
                ax = axes[row, column]
                if column == 0:
                    ax.set_ylabel(f'State: {row + 1}\nAction: {actions[row].type} {actions[row].parameters}')
                for chart_data in chart.data:
                    data = list(zip(*chart_data))
                    ax.plot(x, list(data[1]), color, linewidth=1)
                xs = [0.0, x[-1]]
                ys = [0.0, 1.0]
                for event in chart.events:
                    if event.type is MouseEventType.mousemove or event.type is MouseEventType.mouseover:
                        xs.append(event.x * x[-1] + 500)
                        ys.append(1 - event.y)
                bins = 50
                H = myplot(xs, ys, bins)
                H_masked = np.ma.masked_array(H, H < 3)
                if H_masked.any():
                    x_scale = x[-1] / bins
                    y_scale = 1 / bins
                    origin, width, height = bounding_box(H_masked, x_scale, y_scale)
                    # Create a Rectangle patch
                    rect = Rectangle(origin, width, height, linewidth=1, edgecolor='b', facecolor='none', zorder=2)
                    # Add the patch to the Axes
                    ax.add_patch(rect)
            else:
                axes_to_remove.append(axes[row, column])

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
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
    # plt.savefig(f'fig{i}.png')
    plt.show()


def visualize_eda_session(adapted_patterns: List[Session], patterns: List[Pattern]):
    i = 1
    for adapted_pattern, actions, pattern in zip(adapted_patterns, patterns):
        visualize_state_list(adapted_pattern.states, adapted_pattern.actions, pattern.pattern_class, i)
        i += 1


if __name__ == '__main__':
    with open("../data/deepdrawing/interaction/adapted_patterns.pkl", "rb") as f:
        adapted_patterns: List[Session] = pickle.load(f)

    with open("../data/deepdrawing/interaction/classified_patterns.pkl", "rb") as f:
        patterns: List[Pattern] = pickle.load(f)

    visualize_eda_session(adapted_patterns, patterns)
