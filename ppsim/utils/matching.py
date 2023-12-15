from typing import Iterable, Callable, Any, Optional


def get_filtering_function(user_input: Optional) -> Callable[[Any], bool]:
    """Builds a function f(value) -> bool which says whether that value matches or not the user input.

    :param user_input:
        If None is passed, the function always evaluates to true.
        If an iterable object is passed, checks whether or not the value is in the iterable.
        Otherwise, checks whether or not the value is exactly the user input passed.

    :return:
        The filtering function f(value) -> bool.
    """
    if user_input is None:
        return lambda d: True
    elif isinstance(user_input, Iterable) and not isinstance(user_input, str):
        user_input = set(user_input)
        return lambda d: d in user_input
    else:
        return lambda d: d == user_input


def get_indexed_object(keys: Optional, indexed: dict) -> Optional[list]:
    """Indexes a set of keys in an indexed dictionary.

    :param keys:
        If None is passed, returns None.
        If a single value is passed, returns a list with the single element <indexed[keys]>.
        If an iterable is passed, returns a list with multiple elements <indexed[k]>, with k in keys.

    :param indexed:
        The dictionary to be indexed.

    :return:
        The list of indexed objects, or None in case the keys are not passed.
    """
    if keys is None:
        return None
    keys = [keys] if isinstance(keys, Iterable) and not isinstance(keys, str) else keys
    return [indexed[k] for k in keys]


def get_matching_object(matcher: Optional, index: Any, default: Any):
    """Uses a matching strategy to return a matching object.

    :param matcher:
        If None is passed, returns the default object.
        If a dictionary is passed, returns the object that matches with the given index.
        Otherwise, returns the matcher itself, since it is the same for every index.

    :param index:
        The matching index in case a dictionary matcher is passed.

    :param default:
        The default object in case a None matcher is passed.

    :return:
        The matching object.
    """
    if matcher is None:
        return default
    elif isinstance(matcher, dict):
        matcher = matcher.get(index)
        return default if matcher is None else matcher
    else:
        return matcher
