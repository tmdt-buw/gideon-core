import random

import numpy as np


def _create_training_encoding_length_3():
    training_enc = np.zeros(4)
    training_enc += np.array([2, 1, 1, 0])
    if random.randint(0, 1):
        training_enc += np.array([1, 1, 0, 0])
    else:
        if random.randint(0, 1):
            training_enc += np.array([1, 0, 1, 0])
        else:
            training_enc += np.array([1, 0, 0, 1])
    return training_enc


def _generate_random_training_state():
    ran = random.randint(0, 2)
    if ran == 0:
        training_encoding = np.array([1, 1, 0, 0])
    elif ran == 1:
        training_encoding = np.array([1, 0, 1, 0])
    elif ran == 2:
        training_encoding = np.array([1, 0, 0, 1])
    else:
        training_encoding = 0
    return training_encoding


def create_training_overview_data(amount_of_training_sequences=10):
    training_overview_data = []
    for _ in range(amount_of_training_sequences):
        amount_of_states = random.randint(3, 5)

        training_enc = np.zeros(4)
        training_overview_states = []
        ran = random.randint(0, 1)
        for f_state in range(amount_of_states):
            if f_state == 0:
                if ran:
                    training_enc += np.array([1, 1, 0, 0])
                else:
                    training_enc += np.array([1, 0, 1, 0])
            elif f_state == 1:
                if ran:
                    training_enc += np.array([1, 0, 1, 0])
                else:
                    training_enc += np.array([1, 1, 0, 0])
            elif f_state == 2 or f_state == 3:
                training_enc += _generate_random_training_state()
            elif f_state == 4:
                if random.randint(0, 1):
                    training_enc += _generate_random_training_state()
                else:
                    _fe = _generate_random_training_state()

                    while any((training_enc - _fe) < 0):
                        _fe = _generate_random_training_state()

                    training_enc -= _fe

            training_overview_states.append(training_enc.copy())

        while len(training_overview_states) < 5:
            training_overview_states.append(np.array([-1, -1, -1, -1]))

        training_overview_data.append(training_overview_states)
    training_overview_data = np.array(training_overview_data)
    training_overview_data = np.reshape(training_overview_data, (training_overview_data.shape[0], -1))
    return training_overview_data


def create_training_data(idx, amount_of_normal_sequences=10):
    training_normal_data = []
    encoded_state = [1, 0, 0, 0]
    encoded_state[idx] = 1
    for _ in range(amount_of_normal_sequences):
        amount_of_states = random.randint(3, 5)
        training_enc = np.zeros(4)
        training_normal_states = []
        for f_state in range(amount_of_states):
            if f_state == 0:
                training_enc += _generate_random_training_state()
            elif f_state == 1:
                training_enc += np.array(encoded_state)
            elif f_state == 2:
                training_enc += np.array(encoded_state)
            elif f_state == 3:
                training_enc += _generate_random_training_state()
            elif f_state == 4:
                training_enc += np.array(encoded_state)
            training_normal_states.append(training_enc.copy())
        while len(training_normal_states) < 5:
            training_normal_states.append(np.array([-1, -1, -1, -1]))

        training_normal_data.append(training_normal_states)
    training_normal_data = np.array(training_normal_data)
    training_normal_data = np.reshape(training_normal_data, (training_normal_data.shape[0], -1))
    return training_normal_data
