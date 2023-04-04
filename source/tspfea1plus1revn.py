import numpy as np
import numba
from moptipy.api.process import Process
from moptipy.api.algorithm import Algorithm
from moptipy.examples.tsp.instance import Instance

# np.set_printoptions(threshold=np.inf)


@numba.njit(nogil=True)
def _my_loop_body(i: int,
                  j: int,
                  n: int,
                  matrix: np.ndarray,
                  x: np.ndarray,
                  y: int,
                  y2: int,
                  H: np.ndarray) -> int:
    """Check a move, apply it if it has better H, return new y."""
    im1 = ((i - 1) + n) % n
    jp1 = (j + 1) % n
    """dy is the change of the objective value."""
    dy = -matrix[x[im1]-1][x[i]-1]-matrix[x[j]-1][x[jp1]-1] + \
        +matrix[x[im1]-1][x[j]-1]+matrix[x[i]-1][x[jp1]-1]
    H[y] += 1  # The frequency of objective value of the current solution increases
    y2 = int(y + dy)
    H[y2] += 1  # The frequency of objective value of the new solution increases

    if (H[y2] <= H[y]):
        if (i == 0):
            x[0:j+1:1] = x[j::-1]
        else:
            x[i:j+1:1] = x[j:i-1:-1]  # Reverse the subsequence from i to j in solution x
        return y2
    return y


class MyFeaAlgorithm(Algorithm):
    """An example for a simple FEA algorithm with reversing operator."""

    def __init__(self, ins: Instance) -> None:
        self.name = ins.name
        self.city_number = ins.city_number
        self.cost_tsp = ins.cost_tsp
        self.LB = 0
        self.UB = ins.UB

    def solve(self, process: Process) -> None:
        """
        Solve the problem encapsulated in the provided process.
        This is an optimization process of (1+1) FEA with reversing operator
        :param process: the process instance which provides random numbers,
            functions for creating, copying, and evaluating solutions, as well
            as the termination criterion
        """

        random = process.get_random()
        register = process.register
        should_terminate = process.should_terminate
        ri = random.integers

        H = np.zeros(self.UB, dtype=int)  # H is used to store the access frequency of the objective value

        x = process.create()
        x[:] = range(self.city_number)
        random.shuffle(x)  # randomly generate an initial solution
        y = int(process.evaluate(x))  # get the tour length of this solution
        matrix = self.cost_tsp
        n = self.city_number
        y2 = 0

        while not should_terminate():
            i = ri(n-1)
            j = ri(n-1)
            if i > j:
                i, j = j, i
            if (i == j) or (i == 0 and j == n - 2):
                continue
            y = _my_loop_body(i, j, n, matrix, x, y, y2, H)
            register(x, y)

    def __str__(self):
        """
        Get the name of this algorithm.

        This name is then used in the directory path and file name of the
        log files.

        :returns: fea_revn
        """
        return "fea_revn"
