from typing import List

from model.state import AbstractInteractionData, AbstractState


def is_reset_state(state: AbstractState) -> bool:
    return len(state.charts) == 0


def segment_abstract_data(abstract_interaction_data: AbstractInteractionData) -> List[List[AbstractState]]:
    segments = []
    for abstract_interaction_session in abstract_interaction_data.sessions:
        segment = []
        for abstract_state in abstract_interaction_session.states:
            if is_reset_state(abstract_state):
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append(abstract_state)
        if segment:
            segments.append(segment)
    return segments
