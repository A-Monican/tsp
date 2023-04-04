from moptipy.api.process import Process
from moptipy.api.algorithm import Algorithm
from moptipy.examples.tsp.instance import Instance
import numba  # type: ignore
import numpy as np


@numba.njit(nogil=True)
def _my_loop_body(i: int,
                  j: int,
                  n: int,
                  matrix: np.ndarray,
                  x: np.ndarray,
                  y: int) -> int:
    """Check a move, apply it if it is better, return new y."""
    im1 = ((i - 1) + n) % n
    jp1 = (j + 1) % n

    dy = -matrix[x[im1] - 1][x[i] - 1] - matrix[x[j] - 1][x[jp1] - 1] + \
        +matrix[x[im1] - 1][x[j] - 1] + matrix[x[i] - 1][x[jp1] - 1]
    if dy <= 0:
        if i == 0:
            x[0:j + 1:1] = x[j::-1]
        else:
            x[i:j + 1:1] = x[j:i - 1:-1]  # Reverse the subsequence from i to j in solution x
        return y + dy
    return y


class MyEaAlgorithm(Algorithm):
    """An example for a simple (1+1) EA with reversing operator."""

    def __init__(self, ins: Instance) -> None:
        self.name = ins.name
        self.city_number = ins.city_number
        self.cost_tsp = ins.cost_tsp
        self.LB = 0
        self.UB = ins.UB

    def solve(self, process: Process) -> None:
        """
        Solve the problem encapsulated in the provided process.
        This is an optimization process of (1+1) EA with reversing operator
        :param process: the process instance which provides random numbers,
            functions for creating, copying, and evaluating solutions, as well
            as the termination criterion
        """

        random = process.get_random()
        register = process.register
        should_terminate = process.should_terminate
        ri = random.integers

        x = process.create()
        x[:] = range(self.city_number)
        random.shuffle(x)  # randomly generate an initial solution
        y = process.evaluate(x)  # get the tour length of this solution
        n = self.city_number
        matrix = self.cost_tsp

        while not should_terminate():
            i = ri(n-1)
            j = ri(n-1)
            if i > j:
                i, j = j, i
            if (i == j) or (i == 0 and j == n - 2):
                continue
            y = _my_loop_body(i, j, n, matrix, x, y)
            register(x, y)

    def __str__(self):
        """
        Get the name of this algorithm.

        This name is then used in the directory path and file name of the
        log files.

        :returns: ea_revn
        """
        return "ea_revn"
