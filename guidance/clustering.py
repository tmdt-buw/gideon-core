import pickle
from typing import List, Dict

import numpy as np
from edist import seted
from sklearn.cluster import DBSCAN

from model.state import AbstractState


class Cluster:
    """
    internal cluster data model
    score: avg appearance in original graph
    sequences: all segments that belong to the cluster
    most_rep: most representative segment for the cluster
    """
    score: int
    sequences: List[List[AbstractState]]
    most_rep: List[AbstractState]


def state_distance(state_one: AbstractState, state_two: AbstractState, chart_weight=0.2, label_weight=0.8, dimension_weight=0.8) -> float:
    """
        This function compares two states and returns the (weighted) amount of operations needed to transform one state
        into the other. Each operation can also be weighted. So far the possible operations are:
        - Add or Remove a graph / all graphs
            - represented by the amount of graphs changing in each state
        - Add or remove a dimension of a graph
            - represented by the amount of dimensions changing for each chart of the state
    """

    if not state_one or not state_two:
        return 1

    def scaled_value(value1, value2) -> float:
        if value1 == 0 and value2 == 0:
            return 0
        return max(0.1, abs(value1 - value2) / max(value1, value2))

    def count_labels(state: AbstractState):
        state_stats = {}
        for chart in state.charts:
            for label in chart.labels:
                label_key = f'{label.label_type}_labels'
                if label_key in state_stats:
                    state_stats[f'{label.label_type}_labels'] += len(chart.dimensions)
                else:
                    state_stats[f'{label.label_type}_labels'] = 1

        return state_stats

    # diff of number charts
    count_of_different_charts = scaled_value(len(state_one.charts), len(state_two.charts))
    chart_distance = count_of_different_charts * chart_weight

    # diff in labels
    state_one_stats = count_labels(state_one)
    state_two_stats = count_labels(state_two)
    label_distances = []
    keys = set(state_one_stats.keys())
    keys.update(state_two_stats.keys())
    for key in keys:
        num_state_one = state_one_stats.get(key, 0)
        num_state_two = state_two_stats.get(key, 0)

        label_distances.append(abs((num_state_one / len(state_one.charts)) - (num_state_two / len(state_two.charts))))

    label_distance = sum(label_distances) * label_weight

    # diff in dimensions
    state_one_dimension = set()
    state_one_dimensions = [chart.dimensions for chart in state_one.charts]
    for dimension in state_one_dimensions:
        state_one_dimension.update(dimension)
    state_two_dimension = set()
    state_two_dimensions = [chart.dimensions for chart in state_two.charts]
    for dimension in state_two_dimensions:
        state_two_dimension.update(dimension)
    dimensions_distance = len(state_one_dimension.difference(state_two_dimension)) / max(len(state_one_dimension), len(state_two_dimension)) * dimension_weight

    # put everything together
    distance = chart_distance + label_distance + dimensions_distance
    return distance


def compare_states(first_state: AbstractState, sec_state: AbstractState) -> bool:
    """
        This function compares if two states are equal.
    """
    are_equal = first_state == sec_state
    return are_equal


def filter_states(session: List[AbstractState]):
    states_to_be_kept = [session[0]]
    for idx in range(1, len(session) - 2):
        first_state = states_to_be_kept[-1]
        sec_state = session[idx]
        if not compare_states(first_state, sec_state):
            states_to_be_kept.append(sec_state)

    return states_to_be_kept


def calculate_sequence_edit_distance(first_segment: List[AbstractState], second_segment: List[AbstractState]):
    """
    Calculates the sequence edit distance between two segments
    """
    return seted.seted(first_segment, second_segment, state_distance)


def calculate_distances(interaction_data: List[List[AbstractState]]) -> List[List[float]]:
    """
    Calculates distance matrix between all given sequences
    """
    matrix = np.ones((len(interaction_data), len(interaction_data))) * -1
    for i, segment1 in enumerate(interaction_data):
        for j, segment2 in enumerate(interaction_data):
            matrix[i][j] = calculate_sequence_edit_distance(segment1, segment2)
    return matrix


def cluster_interaction_data(interaction_data: List[List[AbstractState]], number_of_sessions: int, eps: float = 0.3) -> Dict[str, Cluster]:
    """
    Calculates distance matrix and clusters all given sequences using DBSCAN
    """
    df = calculate_distances(interaction_data)
    df_norm = (df - df.min()) / (max(0.000001, df.max() - df.min()))

    clustering = DBSCAN(eps=eps, min_samples=round(number_of_sessions / 3), metric='precomputed').fit(df_norm)
    labels = clustering.labels_

    clusters = {}
    for label in list(set(labels)):
        neighbor_counts = 0

        # build cluster
        cluster = Cluster()
        label_mask = labels == label
        cluster.sequences = np.array(interaction_data)[label_mask].tolist()
        clusters[f'{label}'] = cluster

        # find most representative sequence
        for idx, row in enumerate(df_norm):
            if label_mask[idx]:
                neighbor_mask = row <= eps
                if neighbor_counts < sum(neighbor_mask):
                    neighbor_counts = sum(neighbor_mask)
                    cluster.most_rep = np.array(interaction_data)[idx]

        # calculate representative score for ordering - avg over idx in original session
        score = 0
        for sequence in cluster.sequences:
            score += sequence[0].score
        cluster.score = score / len(cluster.sequences)

    return clusters


if __name__ == '__main__':
    with open("../data/metalarcwelding/interaction/filtered_segments.pkl", "rb") as f:
        filtered_segments = pickle.load(f)

    numer_of_sessions = 4
    eps = 0.2
    segments = []
    for r in range(4):
        for segment in filtered_segments:
            segments.append(segment)
    numer_of_sessions = 3
    clus = cluster_interaction_data(segments, numer_of_sessions, 0.3)
    debug = 1
