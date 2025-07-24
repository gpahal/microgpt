import unicodedata
from typing import cast

import numpy as np
from numpy.typing import NDArray


def _get_counts_dict(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    If an id is -1, it is considered a chunk boundary and not counted.

    Example: ids=[1, 2, 3, -1, 1, 2, 4] -> {(1, 2): 2, (2, 3): 1, (2, 4): 1}

    Args:
        ids: The list of integers to count

    Returns:
        A dictionary of counts of consecutive pairs
    """
    counts_dict: dict[tuple[int, int], int] = {}
    for i in range(len(ids) - 1):
        if ids[i] == -1 or ids[i + 1] == -1:
            continue
        pair = (ids[i], ids[i + 1])
        counts_dict[pair] = counts_dict.get(pair, 0) + 1
    return counts_dict


def _merge_ids(
    ids: list[int],
    pair: tuple[int, int],
    new_id: int,
    counts_dict: dict[tuple[int, int], int],
) -> list[int]:
    if len(ids) < 2:
        return ids

    new_ids = [0] * len(ids)
    j = 0
    i = 0
    while i < len(ids):
        id_i = ids[i]
        if id_i != pair[0] or i == len(ids) - 1:
            new_ids[j] = id_i
            j += 1
            i += 1
            continue

        id_i_next = ids[i + 1]
        if id_i_next != pair[1]:
            new_ids[j] = id_i
            j += 1
            i += 1
            continue

        new_ids[j] = new_id
        j += 1
        if i > 0 and ids[i - 1] != -1:
            curr_pair = (ids[i - 1], ids[i])
            count = counts_dict.get(curr_pair, 0) - 1
            if count <= 0:
                counts_dict.pop(curr_pair, None)
            else:
                counts_dict[curr_pair] = count

            curr_pair = (ids[i - 1], new_id)
            counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1
        if i < len(ids) - 2 and ids[i + 2] != -1:
            curr_pair = (pair[1], ids[i + 2])
            count = counts_dict.get(curr_pair, 0) - 1
            if count <= 0:
                counts_dict.pop(curr_pair, None)
            else:
                counts_dict[curr_pair] = count

            curr_pair = (new_id, ids[i + 2])
            counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1
        ids[i + 1] = new_id
        i += 2

    counts_dict.pop(pair, None)
    return new_ids[:j]


def _merge_ids_tensor(
    ids: NDArray[np.int32],
    pair: tuple[int, int],
    new_id: int,
    counts_dict: dict[tuple[int, int], int],
) -> NDArray[np.int32]:
    """
    In the ids array, replace all consecutive occurrences of pair with the new integer token new_id.
    If an id is -1, it is considered a chunk boundary and not merged.

    Example: ids=[1, 2, 3, 1, 2, 1, -1, 2, 3, 1, 2], pair=(1, 2), new_id=4 -> [4, 3, 4, 1, -1, 2, 3, 4]

    Args:
        ids: The ids array to merge
        pair: The pair of ids to replace
        new_id: The id to replace the pair with
        counts_dict: A dictionary of counts of consecutive pairs. The counts will be updated in place

    Returns:
        A new ids array with the pair replaced with the new integer token new_id
    """
    if len(ids) < 2:
        return ids

    if pair[0] == pair[1]:
        # Remove consecutive indices to avoid overlapping merges
        # For pairs like (1,1), we need to handle consecutive occurrences carefully
        # Example: ids=[1, 1, 1, 1, 1, 2, 1, 1], pair=(1, 1)
        #          -> pair_0_indices=[0, 2, 6] and not [0, 1, 2, 3, 6]

        # Find indices where consecutive occurrences of pair[0] start
        # Example: ids=[1, 1, 1, 1, 1, 2, 1, 1], pair=(1, 1)
        #          -> sequence_indices=[[0, 5], [6, 8]]
        sequence_indices = np.where(np.diff(np.hstack(([False], ids == pair[0], [False]))))[0].reshape(-1, 2)
        if len(sequence_indices) == 0:
            return ids

        # Example: sequence_indices=[[0, 5], [6, 8]]
        #          -> total_pairs=(5-0)//2 + (8-6)//2 = 3
        total_pairs = sum(np.diff(sequence_indices) // 2)

        # Crate pairs from the sequence indices
        # Example: sequence_indices=[[0, 5], [6, 8]], total_pairs=3
        #          -> pairs=[(0, 1), (2, 3), (6, 7)]
        pair_0_indices = np.empty(total_pairs, dtype=np.int32)
        i = 0
        for sequence_index in sequence_indices:
            sequence_length = (sequence_index[1] - sequence_index[0]) // 2
            pair_0_indices[i : i + sequence_length] = np.arange(
                sequence_index[0], sequence_index[0] + sequence_length * 2, 2, dtype=np.int32
            )
            i += sequence_length
    else:
        pair_0_indices = np.where((ids[:-1] == pair[0]) & (ids[1:] == pair[1]))[0]

    if len(pair_0_indices) == 0:
        return ids

    # Update counts_dict
    pair_1_indices = pair_0_indices + 1
    prev_indices = pair_0_indices - 1
    if prev_indices[0] < 0:
        prev_indices[0] = 0
    next_indices = pair_1_indices + 1
    if next_indices[-1] >= len(ids):
        next_indices[-1] = len(ids) - 1
    indices = np.hstack((pair_0_indices[:, None], ids[prev_indices][:, None], ids[next_indices][:, None]))
    indices_list = cast(list[list[int]], indices.tolist())
    next_consecutive_i = -1
    for index_list in indices_list:
        i = index_list[0]
        prev_value = index_list[1]
        next_value = index_list[2]
        if i > 0:
            id_i_prev = new_id if i == next_consecutive_i else prev_value
            if id_i_prev != -1:
                curr_pair = (id_i_prev, pair[0])
                count = counts_dict.get(curr_pair, 0) - 1
                if count <= 0:
                    counts_dict.pop(curr_pair, None)
                else:
                    counts_dict[curr_pair] = count

                curr_pair = (id_i_prev, new_id)
                counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1
        if i < len(ids) - 2:
            id_i_next = next_value
            if id_i_next != -1:
                curr_pair = (pair[1], id_i_next)
                count = counts_dict.get(curr_pair, 0) - 1
                if count <= 0:
                    counts_dict.pop(curr_pair, None)
                else:
                    counts_dict[curr_pair] = count

                curr_pair = (new_id, id_i_next)
                counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1

        next_consecutive_i = i + 2

    counts_dict.pop(pair, None)
    ids[pair_0_indices] = new_id
    ids = np.delete(ids, pair_1_indices)
    return ids


def _replace_control_characters(s: str) -> str:
    """
    Replace control characters with their escaped Unicode representation.
    We don't want to print control characters, which distort the output (e.g. \n or much worse).
    See: https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    See: http://www.unicode.org/reports/tr44/#GC_Values_Table

    Args:
        s: The string to replace control characters in

    Returns:
        A string with control characters replaced with their escaped Unicode representation
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # This character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # Escape
    return "".join(chars)


def _render_token(token: bytes) -> str:
    """
    Pretty print a token, escaping control characters.

    Args:
        token: The token to render

    Returns:
        A string with the token rendered
    """
    s = token.decode("utf-8", errors="replace")
    s = _replace_control_characters(s)
    return s
