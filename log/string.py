from typing import List, Dict

from guidance.clustering import Cluster
from guidance.recurring_pattern import Pattern
from model.state import InteractionData, AbstractInteractionData, AbstractState, State, Session


def print_interaction_data(interaction_data: InteractionData):
    for idx, session in enumerate(interaction_data.sessions):
        print(f"Session {idx}")
        for state in session.states:
            print_interaction_state(state)


def print_interaction_state(state: State):
    string = "["
    for chart in state.charts:
        string += f"{chart.dimensions}|{[label.label_class.severity.value for label in chart.labels]},"
    if state.charts:
        string = string[:-1]
    string += "]"
    print(string)


def print_abstract_interaction_data(interaction_data: AbstractInteractionData):
    for idx, session in enumerate(interaction_data.sessions):
        print(f"Session {idx}")
        for state in session.states:
            string = "["
            for chart in state.charts:
                string += f"[{chart.dimensions}|{[label.label_severity.value + str(label.label_type) for label in chart.labels]}"
            string += "]"
            print(string)


def print_abstract_interaction_sequences(interaction_sequences: List[List[AbstractState]]):
    for idx, sequence in enumerate(interaction_sequences):
        print(f"Sequence {idx}")
        print_abstract_sequence(sequence)


def print_abstract_interaction_sequence_clusters(clusters: Dict[str, Cluster]):
    for key, cluster in clusters.items():
        print(f"Cluster {key}")
        print_abstract_interaction_sequences(cluster.sequences)


def print_abstract_sequence(sequence: List[AbstractState]):
    for state in sequence:
        string = "["
        for chart in state.charts:
            string += f"[{chart.dimensions}|{[label.label_severity.value + str(label.label_type)  for label in chart.labels]}"
        string += "]"
        print(string)


def print_abstract_interaction_cluster_patterns(clusters: Dict[str, Cluster]):
    for key, cluster in clusters.items():
        print(f"Cluster {key}")
        print_abstract_sequence(cluster.most_rep)


def print_abstract_interaction_patterns_with_classes(patterns: List[Pattern]):
    for idx, pattern in enumerate(patterns):
        print(f"Pattern {idx} | Class {pattern.pattern_class}")
        print_abstract_sequence(pattern.pattern)


def print_generated_eda_session(eda_sessions: List[Session]):
    for idx, session in enumerate(eda_sessions):
        print(f"Sequence {idx}")
        for state in session.states:
            print_interaction_state(state)
