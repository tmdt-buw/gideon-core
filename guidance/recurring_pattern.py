from itertools import permutations
from typing import List

import numpy as np

from model.state import AbstractState


class Pattern:
    score: int
    pattern_class: int
    pattern: List[AbstractState]


class Cluster:
    states = []

    def __init__(self, states):
        self.states = states


def search_for_pattern_in_all_states(pattern, states):
    len_s = len(pattern)
    pattern_in_all_states = []
    for state in states:
        pattern_in_state_list = []
        for i in range(len(state) - len_s + 1):
            tmp_pattern = state[i:i + len_s]
            pattern_in_state_list.append(pattern == tmp_pattern)
            _for_debugging = 1
        pattern_in_all_states.append(any(pattern_in_state_list))
    return pattern_in_all_states


def find_longest_contiguous_recurring_pattern_with_permutations(session):
    longest = []
    sequences = {}
    first_states = session.states[0].tolist()
    other_states = [states.tolist() for states in session.states if states.tolist() != first_states]
    for idx in range(len(first_states)):
        for jdx in range(idx, len(first_states) + 1):
            temporary_sequence = first_states[idx:jdx]
            if not temporary_sequence:
                continue

            perms = [list(p) for p in set(permutations(temporary_sequence.copy()))]
            found_in_other_states = []
            for perm in perms:
                found_in_other_states.append(search_for_pattern_in_all_states(perm, other_states))

            dimension_truth = np.array([False for _ in range(len(other_states))])
            for result in found_in_other_states:
                for id_x, val in enumerate(result):
                    if val:
                        dimension_truth[id_x] = True

            res = all(dimension_truth)
            if res:
                if len(temporary_sequence) > len(longest):
                    longest = temporary_sequence.copy()
                if f'{temporary_sequence}' not in sequences:
                    sequences[f'{temporary_sequence}'] = 0
                sequences[f'{temporary_sequence}'] += 1

    return longest, sequences


def find_longest_contiguous_recurring_pattern(session):
    longest = []
    sequences = {}
    first_states = session.states[0].tolist()
    other_states = [states.tolist() for states in session.states if states.tolist() != first_states]
    for idx in range(len(first_states)):
        for jdx in range(idx, len(first_states) + 1):
            temporary_sequence = first_states[idx:jdx]
            if not temporary_sequence:
                continue

            res = search_for_pattern_in_all_states(temporary_sequence, other_states)
            if all(res):
                if len(temporary_sequence) > len(longest):
                    longest = temporary_sequence.copy()
                if f'{temporary_sequence}' not in sequences:
                    sequences[f'{temporary_sequence}'] = 0
                sequences[f'{temporary_sequence}'] += 1

    return longest, sequences


def find_shortest_contiguous_recurring_pattern(session):
    shortest = []
    sequences = {}
    first_states = session.states[0]
    other_states = [states for states in session.states if states != first_states]
    for idx in range(len(first_states)):
        for jdx in range(idx, len(first_states) + 1):
            temporary_sequence = first_states[idx:jdx]
            if not temporary_sequence:
                continue

            res = search_for_pattern_in_all_states(temporary_sequence, other_states)
            if res:
                if not shortest:
                    if 2 <= len(temporary_sequence):
                        shortest = temporary_sequence.copy()
                elif 2 <= len(temporary_sequence) < len(shortest):
                    shortest = temporary_sequence.copy()
                    if f'{temporary_sequence}' not in sequences:
                        sequences[f'{temporary_sequence}'] = 0
                    sequences[f'{temporary_sequence}'] += 1

    return shortest, sequences


def find_start_and_end_of_sequence(wanted_states, all_states):
    ar = []
    max_first_occurrence = 0
    for state in wanted_states:
        g = [i for i, ele in enumerate(all_states) if ele == state]
        ar.append(g)
        if g[0] > max_first_occurrence:
            max_first_occurrence = g[0]

    index_of_minimum = np.inf
    tmp_minimum = np.inf
    for idx, l in enumerate(ar):
        cur_min = min(l)
        if cur_min <= tmp_minimum:
            tmp_minimum = cur_min
            index_of_minimum = idx

    start = index_of_minimum
    all_other_indizes_without_index_containing_current_minimum = [idx for indices in ar for idx in indices
                                                                  if indices != ar[index_of_minimum]]
    if all_other_indizes_without_index_containing_current_minimum:
        for other_value in ar[index_of_minimum]:
            if other_value < min(all_other_indizes_without_index_containing_current_minimum):
                start = other_value

    return start, max_first_occurrence


def find_most_occurred_states(cluster: List[List[AbstractState]]):
    current_highest_count = 0
    current_most_occurred_states = []
    counts = {}
    first_sequence = cluster[0]
    for state in first_sequence:
        # Maybe delete the following line to soften the restriction of being in ALL states.
        if all(state in sec_sequence for sec_sequence in cluster):
            if f'{state}' not in counts:
                some_sum = sum([sec_sequence.count(state) for sec_sequence in cluster])
                occurrence = some_sum
                if occurrence > current_highest_count:
                    current_highest_count = occurrence
                    current_most_occurred_states = [state]
                elif occurrence == current_highest_count:
                    current_most_occurred_states.append(state)

                counts[f'{state}'] = occurrence

    return current_most_occurred_states, current_highest_count, counts


def find_longest_subsequence(cluster: List[List[AbstractState]]):
    current_most_occurred_states, _, c = find_most_occurred_states(cluster)

    seqs = []
    for sequence in cluster:
        sec_start, sec_end = find_start_and_end_of_sequence(current_most_occurred_states, sequence)
        seqs.append(sequence[sec_start:sec_end + 1])
    return seqs


def test_all_functions():
    a = np.array(['Z', 'A', 'B', 'C', 'Z', 'Z', 'Z', 'Z'])
    b = np.array(['A', 'B', 'C', 'Q', 'Q', 'Q', 'A', 'B', 'C'])
    c = np.array(['A', 'C', 'B', 'N', 'N', 'N'])

    # testSession = Cluster([a, b])
    # lon1, dic1 = find_longest_contiguous_recurring_pattern_with_permutations(testSession)
    # sho1, sho_dic1 = find_shortest_contiguous_recurring_pattern(testSession)

    # testSession = Cluster([b, a])
    # lon2, dic2 = find_longest_contiguous_recurring_pattern_with_permutations(testSession)
    # sho2, sho_dic2 = find_shortest_contiguous_recurring_pattern(testSession)

    testSession = Cluster([a, b, c])
    # lon3, dic3 = find_longest_contiguous_recurring_pattern(testSession)
    lon4, dic4 = find_longest_contiguous_recurring_pattern_with_permutations(testSession)
    # sho3, sho_dic3 = find_shortest_contiguous_recurring_pattern(testSession)

    _for_debugging = 1

    """
        Test code to find a subsequence, which includes all the most_wanted states
    """
    # a = ['A', 'Z', 'A', 'B', 'C', 'Z', 'D', 'Z', 'Z', 'A', 'C', 'D', 'B']
    # most_wanted = ['A', 'B', 'C', 'D']
    #
    # s, e = find_start_and_end_of_sequence(most_wanted, a)
    # seq = a[s:e + 1]
    #
    # _for_debugging = 1

    """
        Testcode to find the longest subsequence for all states (a, b, c) in a session.
    """
    a = ['A', 'B', 'C', 'Z', 'D', 'Z', 'Z', 'A', 'C', 'D', 'B']
    b = ['B', 'A', 'N', 'C', 'D', 'N', 'N', 'N']
    c = ['A', 'B', 'N', 'N', 'C']
    # testSession = Cluster([a, b])
    # result1 = find_longest_subsequence(testSession)
    # # expected: [[A, B, C, Z, D], [B, A, N, C, D]]
    #
    # testSession = Cluster([a, b, c])
    # result2 = find_longest_subsequence(testSession)
    # # expected: [[A, B, C], [B, A, N, C], [A, B, N, N, C]]

    _for_debugging = 1


if __name__ == "__main__":
    # with open("../data/interaction_test/clusters.pkl", "rb") as f:
    #     clusters = pickle.load(f)
    #
    # all_sequences = []
    # for key, cluster in clusters.items():
    #     all_states = []
    #     for state in cluster:
    #         all_states.append(state)
    #
    #     if all_states:
    #         session = Cluster(all_states)
    #         seq = find_longest_subsequence(session)
    #         if seq:
    #             all_sequences.append(seq)
    #
    # _for_debugging = 1
    test_all_functions()
