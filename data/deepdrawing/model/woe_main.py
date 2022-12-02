import json
import pickle
from typing import List

import numpy as np
from sklearn.neural_network import MLPClassifier

from data.deepdrawing.model.generator import generate_overview_data_only
from guidance.abstraction import abstract_interaction_data
from guidance.classification import encode_pattern
from guidance.clustering import cluster_interaction_data
from data.deepdrawing.model.training_data import create_training_data, create_training_overview_data
from guidance.filter import filter_segments
from guidance.recurring_pattern import Pattern
from guidance.segmentation import segment_abstract_data
from model.state import LabelClass, Severity

if __name__ == '__main__':
    file = open('../data.json')
    ts_data = json.load(file)
    label_classes = [
        LabelClass(name="clean", severity=Severity.okay),
        LabelClass(name="small", severity=Severity.warning),
        LabelClass(name="large", severity=Severity.error)
    ]
    number_of_sessions = 6
    interaction_data = generate_overview_data_only(ts_data, label_classes, number_of_sessions)

    # Abstract graph
    abstract_data = abstract_interaction_data(interaction_data)

    # Segmentation of data
    segmented_data = segment_abstract_data(abstract_data)

    # filter segments
    filtered_segments = filter_segments(segmented_data)

    # cluster
    clusters = cluster_interaction_data(filtered_segments, number_of_sessions)
    outliers = clusters.pop('-1', None)

    # recurring patterns
    recurring_patterns: List[Pattern] = []
    for key, cluster in clusters.items():
        pattern = Pattern()
        pattern.score = cluster.score
        pattern.pattern = cluster.most_rep
        recurring_patterns.append(pattern)

    with open("trainings_pattern_overview.pkl", "wb") as f:
        pickle.dump(recurring_patterns, f)
    with open("trainings_pattern_overview.pkl", "rb") as f:
        recurring_patterns = pickle.load(f)

    test_encoded_patterns = []
    for sequence in recurring_patterns:
        encoding = encode_pattern(sequence)
        test_encoded_patterns.append(np.reshape(encoding, (20,)))
    test_encoded_patterns = np.array(test_encoded_patterns)

    patterns = []
    targets = []

    overview_data = create_training_overview_data(100)
    normal_data = create_training_data(1, 100)
    warning_data = create_training_data(2, 100)
    error_data = create_training_data(3, 100)

    for idx, d in enumerate([overview_data, normal_data, warning_data, error_data]):
        for val in d:
            patterns.append(val)
            targets.append(idx)

    patterns = np.array(patterns)
    targets = np.array(targets)

    p = np.random.permutation(len(patterns))
    patterns = patterns[p]
    targets = targets[p]

    NN = MLPClassifier(hidden_layer_sizes=8, max_iter=500)
    NN.fit(patterns, targets)
    with open("pattern_classifier.pkl", "wb") as f:
        pickle.dump(NN, f)
    pred = NN.predict(test_encoded_patterns)

    debug = 1
