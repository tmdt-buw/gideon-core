from typing import List

from model.state import AbstractState


def filter_segments(segments: List[List[AbstractState]]) -> List[List[AbstractState]]:
    filtered_data = []
    for segment in segments:
        if len(segment) > 2:
            filtered_data.append(segment)
    return filtered_data
