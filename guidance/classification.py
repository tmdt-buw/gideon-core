import pickle
from typing import List

import numpy as np

from guidance.recurring_pattern import Pattern


def encode_pattern(pattern: Pattern):
    """
    :param pattern: The sequence to encod
    :return: a list of length 5 of all encoded states

    This function transforms a given pattern into its representation as a list of encoded states.
    """
    encoded_states = []
    for state in pattern.pattern[-5:]:
        """
        Each transformed state is represented as 4 integers.
            1. The amount of charts in this state
            2. The sum of all OKAY labels in all graphs
            3. The sum of all WARNING labels in all graphs
            4. The sum of all ERROR labels in all graphs
        """
        encoded = np.zeros(4)
        encoded[0] = len(state.charts)
        for chart in state.charts:
            for label in chart.labels:
                if label.label_severity.name == 'okay':
                    encoded += np.array([0, 1, 0, 0])
                elif label.label_severity.name == 'warning':
                    encoded += np.array([0, 0, 1, 0])
                elif label.label_severity.name == 'error':
                    encoded += (np.array([0, 0, 0, 1]))
        encoded_states.append(encoded)

    while len(encoded_states) < 5:
        encoded_states.append(np.array([-1, -1, -1, -1]))

    return np.array(encoded_states)


def classify_patterns(patterns: List[Pattern]):
    encoded_patterns = []
    for pattern in patterns:
        encoding = encode_pattern(pattern)
        encoded_patterns.append(np.reshape(encoding, (20,)))
    encoded_patterns = np.array(encoded_patterns)
    with open("./data/deepdrawing/model/pattern_classifier.pkl", "rb") as f:
        NN = pickle.load(f)
    for idx, pattern in enumerate(encoded_patterns):
        prediction = NN.predict(pattern.reshape(1, -1))
        patterns[idx].pattern_class = prediction[0]
    return patterns


if __name__ == '__main__':
    with open("../data/deepdrawing/interaction/recurring_patterns.pkl", "rb") as f:
        patterns: List[Pattern] = pickle.load(f)

    classify_patterns(patterns)
