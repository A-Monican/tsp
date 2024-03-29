"""Some internal helper functions and base classes."""

from dataclasses import dataclass
from math import inf
from typing import Any, Final

from moptipy.utils.nputils import rand_seed_check
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, type_error

#: The key for the total number of runs.
KEY_N: Final[str] = "n"

#: The unit of the time axis if time is measured in milliseconds.
TIME_UNIT_MILLIS: Final[str] = "ms"
#: The unit of the time axis of time is measured in FEs
TIME_UNIT_FES: Final[str] = "FEs"

#: The name of the raw objective values data.
F_NAME_RAW: Final[str] = "plainF"
#: The name of the scaled objective values data.
F_NAME_SCALED: Final[str] = "scaledF"
#: The name of the normalized objective values data.
F_NAME_NORMALIZED: Final[str] = "normalizedF"


def check_time_unit(time_unit: Any) -> str:
    """
    Check that the time unit is OK.

    :param time_unit: the time unit
    :return: the time unit string

    >>> check_time_unit("FEs")
    'FEs'
    >>> check_time_unit("ms")
    'ms'
    >>> try:
    ...     check_time_unit(1)
    ... except TypeError as te:
    ...     print(te)
    time_unit should be an instance of str but is int, namely '1'.
    >>> try:
    ...     check_time_unit("blabedibla")
    ... except ValueError as ve:
    ...     print(ve)
    Invalid time unit 'blabedibla', only 'FEs' and 'ms' are permitted.
    """
    if not isinstance(time_unit, str):
        raise type_error(time_unit, "time_unit", str)
    if time_unit in (TIME_UNIT_FES, TIME_UNIT_MILLIS):
        return time_unit
    raise ValueError(
        f"Invalid time unit {time_unit!r}, only {TIME_UNIT_FES!r} "
        f"and {TIME_UNIT_MILLIS!r} are permitted.")


def check_f_name(f_name: Any) -> str:
    """
    Check whether an objective value name is valid.

    :param f_name: the name of the objective function dimension
    :return: the name of the objective function dimension

    >>> check_f_name("plainF")
    'plainF'
    >>> check_f_name("scaledF")
    'scaledF'
    >>> check_f_name("normalizedF")
    'normalizedF'
    >>> try:
    ...     check_f_name(1.0)
    ... except TypeError as te:
    ...     print(te)
    f_name should be an instance of str but is float, namely '1.0'.
    >>> try:
    ...     check_f_name("oops")
    ... except ValueError as ve:
    ...     print(ve)
    Invalid f name 'oops', only 'plainF', 'scaledF', and 'normalizedF' \
are permitted.
    """
    if not isinstance(f_name, str):
        raise type_error(f_name, "f_name", str)
    if f_name in (F_NAME_RAW, F_NAME_SCALED, F_NAME_NORMALIZED):
        return f_name
    raise ValueError(
        f"Invalid f name {f_name!r}, only {F_NAME_RAW!r}, "
        f"{F_NAME_SCALED!r}, and {F_NAME_NORMALIZED!r} are permitted.")


@dataclass(frozen=True, init=False, order=True)
class PerRunData:
    """
    An immutable record of information over a single run.

    >>> p = PerRunData("a", "i", 234)
    >>> print(p.instance)
    i
    >>> print(p.algorithm)
    a
    >>> print(p.rand_seed)
    234
    """

    #: The algorithm that was applied.
    algorithm: str

    #: The problem instance that was solved.
    instance: str

    #: The seed of the random number generator.
    rand_seed: int

    def __init__(self, algorithm: str, instance: str, rand_seed: int):
        """
        Create a per-run data record.

        :param str algorithm: the algorithm name
        :param str instance: the instance name
        :param int rand_seed: the random seed
        """
        if not isinstance(algorithm, str):
            raise type_error(algorithm, "algorithm", str)
        if algorithm != sanitize_name(algorithm):
            raise ValueError(f"Invalid algorithm name {algorithm!r}.")
        object.__setattr__(self, "algorithm", algorithm)

        if not isinstance(instance, str):
            raise type_error(instance, "instance", str)
        if instance != sanitize_name(instance):
            raise ValueError(f"Invalid instance name {instance!r}.")
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "rand_seed", rand_seed_check(rand_seed))


@dataclass(frozen=True, init=False, order=True)
class MultiRunData:
    """
    A class that represents statistics over a set of runs.

    If one algorithm*instance is used, then `algorithm` and `instance` are
    defined. Otherwise, only the parameter which is the same over all recorded
    runs is defined.

    >>> p = MultiRunData("a", "i", 3)
    >>> print(p.instance)
    i
    >>> print(p.algorithm)
    a
    >>> print(p.n)
    3
    """

    #: The algorithm that was applied, if the same over all runs.
    algorithm: str | None
    #: The problem instance that was solved, if the same over all runs.
    instance: str | None
    #: The number of runs over which the statistic information is computed.
    n: int

    def __init__(self, algorithm: str | None, instance: str | None, n: int):
        """
        Create the dataset of an experiment-setup combination.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm
        :param instance: the instance name, if all runs are on the same
            instance
        :param n: the total number of runs
        """
        if (algorithm is not None) and \
                (algorithm != sanitize_name(algorithm)):
            raise ValueError(f"Invalid algorithm {algorithm!r}.")
        object.__setattr__(self, "algorithm", algorithm)

        if (instance is not None) and (instance != sanitize_name(instance)):
            raise ValueError(f"Invalid instance {instance!r}.")
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "n", check_int_range(n, "n", 1))


@dataclass(frozen=True, init=False, order=True)
class MultiRun2DData(MultiRunData):
    """
    A multi-run data based on one time and one objective dimension.

    >>> p = MultiRun2DData("a", "i", 3, TIME_UNIT_FES, F_NAME_SCALED)
    >>> print(p.instance)
    i
    >>> print(p.algorithm)
    a
    >>> print(p.n)
    3
    >>> print(p.time_unit)
    FEs
    >>> print(p.f_name)
    scaledF
    """

    #: The unit of the time axis.
    time_unit: str
    #: the name of the objective value axis.
    f_name: str

    def __init__(self, algorithm: str | None, instance: str | None, n: int,
                 time_unit: str, f_name: str):
        """
        Create multi-run data based on one time and one objective dimension.

        :param algorithm: the algorithm name, if all runs are with the same
            algorithm
        :param instance: the instance name, if all runs are on the same
            instance
        :param n: the total number of runs
        :param time_unit: the time unit
        :param f_name: the objective dimension name
        """
        super().__init__(algorithm, instance, n)
        object.__setattr__(self, "time_unit", check_time_unit(time_unit))
        object.__setattr__(self, "f_name", check_f_name(f_name))


def get_instance(obj: PerRunData | MultiRunData) -> str | None:
    """
    Get the instance of a given object.

    :param obj: the object
    :return: the instance string, or `None` if no instance is specified

    >>> p1 = MultiRunData("a", "i1", 3)
    >>> print(get_instance(p1))
    i1
    >>> p2 = PerRunData("a", "i2", 31)
    >>> print(get_instance(p2))
    i2
    """
    return obj.instance


def get_algorithm(obj: PerRunData | MultiRunData) -> str | None:
    """
    Get the algorithm of a given object.

    :param obj: the object
    :return: the algorithm string, or `None` if no algorithm is specified

    >>> p1 = MultiRunData("a1", "i1", 3)
    >>> print(get_algorithm(p1))
    a1
    >>> p2 = PerRunData("a2", "i2", 31)
    >>> print(get_algorithm(p2))
    a2
    """
    return obj.algorithm


def sort_key(obj: PerRunData | MultiRunData) -> \
        tuple[str, str, str, int, int, str, str, float]:
    """
    Get a default sort key for the given object.

    The sort key is a tuple with well-defined field elements that should
    allow for a default and consistent sorting over many different elements of
    the experiment evaluation data API. Sorting should work also for lists
    containing elements of different classes.

    :param obj: the object
    :return: the sort key

    >>> p1 = MultiRunData("a1", "i1", 3)
    >>> p2 = PerRunData("a2", "i2", 31)
    >>> print(sort_key(p1) < sort_key(p2))
    True
    >>> print(sort_key(p1) >= sort_key(p2))
    False
    >>> p3 = MultiRun2DData("a", "i", 3, TIME_UNIT_FES, F_NAME_SCALED)
    >>> print(sort_key(p3) < sort_key(p1))
    True
    >>> print(sort_key(p3) >= sort_key(p1))
    False
    """
    goal_f = getattr(obj, "goal_f") if hasattr(obj, "goal_f") else None
    if goal_f is None:
        goal_f = inf

    return obj.__class__.__name__, \
        "" if obj.algorithm is None else obj.algorithm, \
        "" if obj.instance is None else obj.instance, \
        obj.n if isinstance(obj, MultiRunData) else 0, \
        obj.rand_seed if isinstance(obj, PerRunData) else 0, \
        obj.time_unit if isinstance(obj, MultiRun2DData) else "", \
        obj.f_name if isinstance(obj, MultiRun2DData) else "", \
        goal_f
