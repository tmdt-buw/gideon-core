import pickle
from datetime import datetime
from typing import List

import pandas as pd

from model.state import InteractionData, Session, Chart, State, Label, AbstractLabel, AbstractPanZoom, PanZoom, AbstractInteractionData, AbstractSession, AbstractState, AbstractChart, Severity

dimension_dict = {}
labels_dict = {
    "Normal": 0
}


def encode_dimensions(dimensions):
    """
    Encodes dimensions by replacing the dimension with a number
    """
    encoded = []
    for dimension in dimensions:
        if dimension not in dimension_dict.keys():
            dimension_dict[dimension] = len(dimension_dict.keys())
        encoded.append(dimension_dict[dimension])
    return encoded


def encode_label(label):
    """
    Encodes labels by replacing the label with a number
    """
    key = label.label_class.name
    if key not in labels_dict.keys():
        labels_dict[key] = len(labels_dict.keys())
    return labels_dict[key]


def abstract_chart_pan_zoom(pan_zoom: PanZoom, chart_range: [float, float, float, float]) -> AbstractPanZoom:
    """
    Converts absolute coordinates to relative values
    """
    abstract_pan_zoom = AbstractPanZoom(
        x=0,
        y=0,
        zoom=pan_zoom.zoom
    )
    if pan_zoom.x > 0:
        abstract_pan_zoom.x = (pan_zoom.x - chart_range[0]) / (chart_range[1] - chart_range[0])
    if pan_zoom.y > 0:
        abstract_pan_zoom.y = (pan_zoom.y - chart_range[2]) / (chart_range[3] - chart_range[2])
    return abstract_pan_zoom


def abstract_chart_labels(labels: List[Label], dimensions: List[int], chart_range: [float, float, float, float]) -> List[AbstractLabel]:
    """
    Abstracts all labels replacing absolute with relative values and encoding data specific values
    """
    abstract_labels = []
    if len(labels) > 0:
        for label in labels:
            abstract_label = AbstractLabel(
                label_type=encode_label(label),
                label_severity=label.label_class.severity,
                start=(label.start - chart_range[0]) / (chart_range[1] - chart_range[0]),
                end=(label.end - chart_range[0]) / (chart_range[1] - chart_range[0]),
                dimensions=[dimension_dict.get(dimension) for dimension in label.dimensions]
            )
            abstract_labels.append(abstract_label)
    else:
        okay_label = AbstractLabel(
            label_type=0,
            label_severity=Severity.okay,
            start=0,
            end=1,
            dimensions=dimensions
        )
        abstract_labels.append(okay_label)
    return abstract_labels


def abstract_state_chart(chart: Chart) -> AbstractChart:
    values = []
    for series in chart.data:
        values += series
    x_values = [float(data[0]) for data in values]
    y_values = [data[1] for data in values]
    chart_range = [min(x_values), max(x_values), min(y_values), max(y_values)]
    abstract_chart = AbstractChart(
        dimensions=encode_dimensions(chart.dimensions),
        data_distances=[],
        labels=abstract_chart_labels([label for label in chart.labels if label.start >= chart_range[0] or label.end <= chart_range[1]], encode_dimensions(chart.dimensions), chart_range),
        panZoom=abstract_chart_pan_zoom(chart.panZoom, chart_range)
    )
    return abstract_chart


def calc_active_time_for_charts(charts: List[Chart]) -> int:
    if not charts:
        return 0
    times = [datetime.fromtimestamp(int(event.time / 1000)) for chart in charts for event in chart.events]
    if not times:
        return 0
    times.sort()
    df = pd.DataFrame({'dates': times})
    count = df['dates'].round('100ms').nunique()
    return count


def abstract_session_state(idx: int, state: State) -> AbstractState:
    abstract_state = AbstractState(
        ref=state.id,
        score=idx,
        active_time=calc_active_time_for_charts(state.charts)
    )
    for chart in state.charts:
        abstract_state.charts.append(abstract_state_chart(chart))
    return abstract_state


def abstract_interaction_session(interaction_session: Session) -> AbstractSession:
    abstract_session = AbstractSession()
    for idx, state in enumerate(interaction_session.states):
        abstract_session.states.append(abstract_session_state(idx, state))
    return abstract_session


def abstract_interaction_data(interaction_data: InteractionData) -> AbstractInteractionData:
    abstract_data = AbstractInteractionData()
    for session in interaction_data.sessions:
        abstract_data.sessions.append(abstract_interaction_session(session))
    return abstract_data


if __name__ == '__main__':
    with open("../data/deepdrawing/interaction/interaction_data.pkl", "rb") as f:
        interaction_data = pickle.load(f)

    # Abstract graph
    print("test")
    abstract_data = abstract_interaction_data(interaction_data)
