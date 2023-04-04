from moptipy.api.process import Process
from moptipy.api.algorithm import Algorithm
from moptipy.examples.tsp.instance import Instance
import numba  # type: ignore
import numpy as np


@numba.njit(nogil=True)
def _my_loop_body(i: int,
                  j: int,
                  u: np.float64,
                  n: int,
                  Ts: int,
                  a: np.float64,
                  tau: int,
                  matrix: np.ndarray,
                  x: np.ndarray,
                  y: int,
                  y2: int,
                  H: np.ndarray) -> int:
    """Check a move, apply it if it is better, return new y."""
    im1 = ((i - 1) + n) % n
    jp1 = (j + 1) % n

    tau += 1  # current iteration number
    t = Ts * (1 - a) ** (tau - 1)  # current temperature
    """dy is the change of the objective value."""
    dy = -matrix[x[im1] - 1][x[i] - 1] - matrix[x[j] - 1][x[jp1] - 1] + \
        +matrix[x[im1] - 1][x[j] - 1] + matrix[x[i] - 1][x[jp1] - 1]
    H[y] += 1  # The frequency of objective value of the current solution increases
    y2 = int(y + dy)
    H[y2] += 1  # The frequency of objective value of the new solution increases

    dH = H[y2] - H[y]

    p = np.exp(- dH / t)  # Calculate the current acceptance probability

    if (u < p):
        if i == 0:
            x[0:j + 1:1] = x[j::-1]
        else:
            x[i:j + 1:1] = x[j:i - 1:-1]  # Reverse the subsequence from i to j in solution x
        return y2
    return y


class MyFsaAlgorithm(Algorithm):
    """An example for a simple FSA algorithm with reversing operator."""

    def __init__(self, ins: Instance) -> None:
        self.name = ins.name
        self.city_number = ins.city_number
        self.cost_tsp = ins.cost_tsp
        self.LB = 0
        self.UB = ins.UB

    def solve(self, process: Process) -> None:
        """
        Solve the problem encapsulated in the provided process.
        This is an optimization process of (1+1) FSA with reversing operator
        :param process: the process instance which provides random numbers,
            functions for creating, copying, and evaluating solutions, as well
            as the termination criterion
        """

        random = process.get_random()
        register = process.register
        should_terminate = process.should_terminate
        ri = random.integers
        rj = random.random

        H = np.zeros(self.UB, dtype=int)  # H is used to store the access frequency of the objective value

        x = process.create()
        x[:] = range(self.city_number)
        random.shuffle(x)  # randomly generate an initial solution
        y = int(process.evaluate(x))  # get the tour length of this solution
        n = self.city_number
        matrix = self.cost_tsp
        y2 = 0

        Ts = 2 # Set the starting temperature
        a = 1 - (1 / Ts)**(1 / 10000000000)  # The cooling rate is according to the number of iterations, 10B here
        # a = 1 - (1 / Ts)**(1 / 100000000)  # 100M
        # a = 0.000008
        tau = 0

        while not should_terminate():
            i = ri(n-1)
            j = ri(n-1)
            if i > j:
                i, j = j, i
            if (i == j) or (i == 0 and j == n - 2):
                continue
            u = rj()
            y = _my_loop_body(i, j, u, n, Ts, a, tau, matrix, x, y, y2, H)
            register(x, y)

    def __str__(self):
        """
        Get the name of this algorithm.

        This name is then used in the directory path and file name of the
        log files.

        :returns: fsa_revn
        """
        return "fsa_revn"
