import pickle
from typing import List

from guidance.recurring_pattern import Pattern
from log.string import print_generated_eda_session
from model.state import InteractionData, Session


def adapt_patterns_to_original_data_set_and_sort(patterns: List[Pattern], interaction_data: InteractionData) -> List[Session]:
    state_dict = {}
    action_dict = {}
    for session in interaction_data.sessions:
        for state in session.states:
            state_dict[state.id] = state
        for action in session.actions:
            action_dict[action.to] = action
    result = []
    patterns.sort(key=lambda x: x.score)
    for pattern in patterns:
        session = Session()
        for abstract_state in pattern.pattern:
            session.states.append(state_dict[abstract_state.ref])
            session.actions.append(action_dict[abstract_state.ref])
        result.append(session)
    return result


if __name__ == '__main__':
    with open("../data/deepdrawing/interaction/interaction_data.pkl", "rb") as f:
        interaction_data: InteractionData = pickle.load(f)

    with open("../data/deepdrawing/interaction/classified_patterns.pkl", "rb") as f:
        patterns: List[Pattern] = pickle.load(f)

    adapted_patterns = adapt_patterns_to_original_data_set_and_sort(patterns, interaction_data)
    print_generated_eda_session(adapted_patterns)
