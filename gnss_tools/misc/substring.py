


def find_substring_in_list(
    strings: list[str],
    substr: str,
) -> int | None:
    """
    Find the index of the first occurrence of substr in the list of strings.
    """
    for i, string in enumerate(strings):
        if string.find(substr) != -1:
            return i
    return None

