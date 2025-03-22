import unicodedata

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


def _merge_ids_small(
    ids: NDArray[np.int32],
    pair: tuple[int, int],
    new_id: int,
    counts_dict: dict[tuple[int, int], int],
) -> NDArray[np.int32]:
    if len(ids) < 2:
        return ids

    ids = ids.tolist()
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
                counts_dict.pop(curr_pair)
            else:
                counts_dict[curr_pair] = count

            curr_pair = (ids[i - 1], new_id)
            counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1
        if i < len(ids) - 2 and ids[i + 2] != -1:
            curr_pair = (pair[1], ids[i + 2])
            count = counts_dict.get(curr_pair, 0) - 1
            if count <= 0:
                counts_dict.pop(curr_pair)
            else:
                counts_dict[curr_pair] = count

            curr_pair = (new_id, ids[i + 2])
            counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1
        ids[i + 1] = new_id
        i += 2

    counts_dict.pop(pair, None)
    return np.array(new_ids[:j], dtype=np.int32)


def _merge_ids(
    ids: NDArray[np.int32],
    pair: tuple[int, int],
    new_id: int,
    counts_dict: dict[tuple[int, int], int],
) -> NDArray[np.int32]:
    """
    In the ids array, replace all consecutive occurrences of pair with the new integer token new_idx.
    If an id is -1, it is considered a chunk boundary and not merged.

    Example: ids=[1, 2, 3, 1, 2, 1, -1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4, 1, -1, 2, 3, 4]

    Args:
        ids: The ids array to merge
        pair: The pair of ids to replace
        new_id: The id to replace the pair with
        counts_dict: A dictionary of counts of consecutive pairs. The counts will be updated in place

    Returns:
        A new ids array with the pair replaced with the new integer token new_idx
    """
    if len(ids) <= 32:
        return _merge_ids_small(ids=ids, pair=pair, new_id=new_id, counts_dict=counts_dict)

    if pair[0] == pair[1]:
        # Remove consecutive indices to avoid overlapping merges
        # For pairs like (1,1), we need to handle consecutive occurrences carefully
        # Example: ids=[1, 1, 1, 1, 1, 2, 1, 1], pair=(1, 1)
        #          -> pair_0_indices=[0, 2, 6] and not [0, 1, 2, 3, 6]

        # Find indices where consecutive occurrences of pair[0] start
        # Example: ids=[1, 1, 1, 1, 1, 2, 1, 1], pair=(1, 1)
        #          -> sequence_indices=[[0, 5], [6, 8]]
        sequence_indices = np.where(np.diff(np.hstack(([False], ids == pair[0], [False]))))[
            0
        ].reshape(-1, 2)
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
    next_consecutive_i = -1
    for i in pair_0_indices.tolist():
        if i > 0:
            id_i_prev = new_id if i == next_consecutive_i else int(ids[i - 1])
            if id_i_prev != -1:
                curr_pair = (id_i_prev, pair[0])
                count = counts_dict.get(curr_pair, 0) - 1
                if count <= 0:
                    counts_dict.pop(curr_pair)
                else:
                    counts_dict[curr_pair] = count

                curr_pair = (id_i_prev, new_id)
                counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1
        if i < len(ids) - 2:
            id_i_next = int(ids[i + 2])
            if id_i_next != -1:
                curr_pair = (pair[1], id_i_next)
                count = counts_dict.get(curr_pair, 0) - 1
                if count <= 0:
                    counts_dict.pop(curr_pair)
                else:
                    counts_dict[curr_pair] = count

                curr_pair = (new_id, id_i_next)
                counts_dict[curr_pair] = counts_dict.get(curr_pair, 0) + 1

        next_consecutive_i = i + 2

    counts_dict.pop(pair, None)
    ids[pair_0_indices] = new_id
    ids = np.delete(ids, pair_0_indices + 1)
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


def _render_token(t: bytes) -> str:
    """
    Pretty print a token, escaping control characters.

    Args:
        t: The token to render

    Returns:
        A string with the token rendered
    """
    s = t.decode("utf-8", errors="replace")
    s = _replace_control_characters(s)
    return s
