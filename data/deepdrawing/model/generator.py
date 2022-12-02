import copy
import datetime
import json
import random
from typing import List, Dict
from uuid import uuid4 as uuid

import numpy as np
from tslearn.metrics import cdist_dtw

from model.mouse_event_record import MouseEventRecord, MouseEventType
from model.state import InteractionData, Session, State, Chart, Label, LabelClass, PanZoom, Severity


def _generate_noise(ts_data: List[List[float]], state: State) -> List[State]:
    state_add = copy.deepcopy(state)
    state_remove = copy.deepcopy(state)
    for chart in state_add.charts:
        chart.events = []
    for chart in state_remove.charts:
        chart.events = []
    state_add.id = uuid()
    state_remove.id = uuid()
    chart = Chart(
        dimensions=[1],
        data=list(random.choice(ts_data)),
        labels=[],
        panZoom=PanZoom(),
        events=generate_mouse_noise()
    )
    state_add.charts.append(chart)
    return [state_add, state_remove]


def generate_noise(ts_data: Dict[str, List[List[float]]]) -> List[State]:
    all_data = []
    for key in ts_data.keys():
        all_data += ts_data[key]
    result = [State()]
    result += _generate_noise(all_data, result[-1])
    return result


def generate_noise_for_sequence(ts_data: List[List[float]], states: List[State]):
    if random.random() > 0.1:
        idx = random.randint(0, len(states) - 1)
        for i, el in enumerate(_generate_noise(ts_data, states[idx])):
            states.insert(idx, el)
        return states
    else:
        return states


def generate_class_overview(ts_data: Dict[str, List[List[float]]], label_classes: List[LabelClass]) -> List[State]:
    charts: List[Chart] = []
    states: List[State] = []
    for i, label_class in enumerate(label_classes):
        data = ts_data[label_class.name]
        idx = random.randint(0, len(data) - 1)
        series = data[idx]
        chart = Chart(
            dimensions=[1],
            data=series,
            labels=generate_labels(idx, data[idx], label_class),
            panZoom=PanZoom()
        )
        charts.append(chart)
        state = State(charts=copy.deepcopy(charts.copy()))
        if i == len(label_classes) - 1:
            for chart in state.charts:
                chart.events = generate_mouse_interactions(chart.data)
        else:
            for chart in state.charts:
                chart.events = generate_mouse_noise()
        states.append(state)
    all_data = list(np.concatenate([ts_data["clean"], ts_data["small"], ts_data["large"]]))
    states = generate_noise_for_sequence(all_data, states)
    return states


def generate_variants_of_dataset(n: int, ts_data: List[List[float]]) -> [int]:
    distance_matrix = cdist_dtw(ts_data, n_jobs=-1)
    matrix_length = len(distance_matrix[0]) - 1
    current_best = [0, 0, 0]
    current_best_distance = 0
    for i in range(round(n * matrix_length / 2)):
        idx1 = random.randint(0, matrix_length)
        idx2 = random.randint(0, matrix_length)
        idx3 = random.randint(0, matrix_length)
        total_distance = distance_matrix[idx1][idx2] + distance_matrix[idx1][idx3] + distance_matrix[idx2][idx3]
        if total_distance > current_best_distance:
            current_best_distance = total_distance
            current_best = [idx1, idx2, idx3]
    return current_best


def generate_labels(sample: int, series: List[float], label_class: LabelClass) -> List[Label]:
    labels = []
    label_type = label_class.severity
    if label_type is not Severity.okay:
        labels.append(Label(
            label_class=label_class,
            start=300,
            end=400,
            sample=sample,
            dimensions=[1]
        ))
    return labels


def generate_variations(ts_data: Dict[str, List[List[float]]], label_class: LabelClass) -> List[State]:
    n = random.randint(3, 5)
    series = ts_data[label_class.name]
    idxs = generate_variants_of_dataset(n, series)
    charts: List[Chart] = []
    states: List[State] = [State()]
    # create sequence
    for i, idx in enumerate(idxs):
        serie = series[idx]
        events = generate_mouse_noise()
        if i == len(idxs) - 1:
            events = generate_mouse_interactions(serie)
        chart = Chart(
            dimensions=[1],
            data=serie,
            labels=generate_labels(idx, serie, label_class),
            panZoom=PanZoom(),
            events=events
        )
        charts.append(chart)
        state = State(charts=copy.deepcopy(charts.copy()))
        if i == len(idxs) - 1:
            for chart in state.charts:
                chart.events = generate_mouse_interactions(chart.data)
        else:
            for chart in state.charts:
                chart.events = generate_mouse_noise()
        states.append(state)
    all_data = list(np.concatenate([ts_data["clean"], ts_data["small"], ts_data["large"]]))
    states = generate_noise_for_sequence(all_data, states)
    return states


def generate_mouse_interactions(chart_data: List[float]) -> List[MouseEventRecord]:
    events = []
    date = datetime.datetime.now()
    for i in range(100):
        xidx = random.randint(350, 400)
        event = MouseEventRecord(
            time=date.timestamp(),
            x=xidx / 420,
            y=chart_data[xidx],
            type=MouseEventType.mousemove,
            element='#chart'
        )
        date = date + datetime.timedelta(seconds=0.1)
        events.append(event)
    return events


def generate_mouse_noise() -> List[MouseEventRecord]:
    events = []
    date = datetime.datetime.now()
    for i in range(10):
        event = MouseEventRecord(
            time=date.timestamp(),
            x=random.randint(0, 420) / 420,
            y=random.random(),
            type=MouseEventType.mousemove,
            element='#chart'
        )
        date = date + datetime.timedelta(seconds=3)
        events.append(event)
    return events


def generate_interaction_session(ts_data: Dict[str, List[List[float]]], label_classes: List[LabelClass]) -> Session:
    session = Session()
    states = []
    states += generate_noise(ts_data)
    states += generate_class_overview(ts_data, label_classes)
    for label_class in label_classes:
        states += generate_variations(ts_data, label_class)
        states += generate_noise(ts_data)
    for state in states:
        if session.states and session.states[-1].charts or state.charts:
            session.states.append(state)
    return session


def generate_interaction_session_overview_only(ts_data: Dict[str, List[List[float]]], label_classes: List[LabelClass]) -> Session:
    session = Session()
    states = []
    states += generate_noise(ts_data)
    states += generate_class_overview(ts_data, label_classes)
    for state in states:
        if session.states and session.states[-1].charts or state.charts:
            session.states.append(state)
    return session


def generate_interaction_data(ts_data: Dict[str, List[List[float]]], label_classes: List[LabelClass], number_of_sessions: int) -> InteractionData:
    interaction_data = InteractionData()
    for i in range(number_of_sessions):
        interaction_data.sessions.append(generate_interaction_session(ts_data, label_classes))
    return interaction_data


def generate_overview_data_only(ts_data: Dict[str, List[List[float]]], label_classes: List[LabelClass], number_of_sessions: int) -> InteractionData:
    interaction_data = InteractionData()
    for i in range(number_of_sessions):
        interaction_data.sessions.append(generate_interaction_session_overview_only(ts_data, label_classes))
    return interaction_data
