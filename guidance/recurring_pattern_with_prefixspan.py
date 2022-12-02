import pickle

import prefixspan


def compareStates(one, two):
    return one == two


if __name__ == '__main__':
    with open("../data/interaction_test/clusters.pkl", "rb") as f:
        clusters = pickle.load(f)

    testcluster = clusters['0']
    # for cluster in clusters:
    all_sequences_as_integer = []
    for sequence in testcluster:
        states_as_integer = []
        seen_equal_states = []
        for state in sequence:
            index = -1
            for idx, equal_states in enumerate(seen_equal_states):
                # if state in equal_states:
                #     index = idx
                #     equal_states.append(equal_states)
                for equal_state in equal_states:
                    if compareStates(state, equal_state):
                        index = idx
                        equal_states.append(equal_states)

            if index == -1:
                seen_equal_states.append([state])
                key = len(seen_equal_states) - 1
            else:
                seen_equal_states[index].append(state)
                key = index

            states_as_integer.append(key)
        all_sequences_as_integer.append(states_as_integer)

    ps = prefixspan.PrefixSpan(all_sequences_as_integer)
    print(ps.topk(5, filter=lambda patt, matches: len(patt) > 1))

    debug = 1
