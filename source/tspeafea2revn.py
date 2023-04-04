import numba
import numpy as np

from moptipy.api.algorithm import Algorithm
from moptipy.api.process import Process
from moptipy.examples.tsp.instance import Instance


@numba.njit(nogil=True)
def _my_ea_loop_body(i: int,
                     j: int,
                     n: int,
                     matrix: np.ndarray,
                     xc: np.ndarray,
                     yc: int) -> int:
    """Check a move, apply it if it is better, return new y."""
    im1 = ((i - 1) + n) % n
    jp1 = (j + 1) % n
    """dy is the change of the objective value."""
    dy = -matrix[xc[im1] - 1][xc[i] - 1] - matrix[xc[j] - 1][xc[jp1] - 1] + \
        +matrix[xc[im1] - 1][xc[j] - 1] + matrix[xc[i] - 1][xc[jp1] - 1]
    if dy <= 0:
        if i == 0:
            xc[0:j + 1:1] = xc[j::-1]
        else:
            xc[i:j + 1:1] = xc[j:i - 1:-1]  # Reverse the subsequence from i to j in solution x
        return int(yc + dy)
    return yc


@numba.njit(nogil=True)
def _my_fea_loop_body(i: int,
                      j: int,
                      n: int,
                      matrix: np.ndarray,
                      xd: np.ndarray,
                      yd: int,
                      H: np.ndarray) -> int:
    """Check a move, apply it if it has better H, return new y."""
    im1 = ((i - 1) + n) % n
    jp1 = (j + 1) % n
    dy = -matrix[xd[im1] - 1][xd[i] - 1] - matrix[xd[j] - 1][xd[jp1] - 1] + \
        +matrix[xd[im1] - 1][xd[j] - 1] + matrix[xd[i] - 1][xd[jp1] - 1]
    H[yd] += 1  # The frequency of objective value of the current solution increases
    y2 = int(yd + dy)
    H[y2] += 1  # The frequency of objective value of the new solution increases

    if H[y2] <= H[yd]:
        if i == 0:
            xd[0:j + 1:1] = xd[j::-1]
        else:
            xd[i:j + 1:1] = xd[j:i - 1:-1]  # Reverse the subsequence from i to j in solution x
        return y2
    return yd


class MyEafea2Algorithm(Algorithm):
    """An example for a EAFEA(A) hybridization algorithm."""

    def __init__(self, ins: Instance) -> None:
        self.name = ins.name
        self.city_number = ins.city_number
        self.cost_tsp = ins.cost_tsp
        self.LB = 0
        self.UB = ins.UB

    def solve(self, process: Process) -> None:
        """
        Solve the problem encapsulated in the provided process.
        This is an optimization process of EAFEA(A) hybridization algorithm based on EA algorithm with reversing operator.
        :param process: the process instance which provides random numbers,
            functions for creating, copying, and evaluating solutions, as well
            as the termination criterion
        """

        random = process.get_random()
        register = process.register
        should_terminate = process.should_terminate
        ri = random.integers

        H = np.zeros(self.UB, dtype=int)  # H is used to store the access frequency of the objective value

        xc = process.create()
        xc[:] = range(self.city_number)
        random.shuffle(xc)  # randomly generate an initial solution
        y = process.evaluate(xc)  # get the tour length of this solution
        matrix = self.cost_tsp
        n = self.city_number
        useFFA = True  # flag of useFFA or not
        xd = xc.copy()
        yc = int(y)
        yd = int(y)
        nm1 = n - 1
        nm2 = n - 2

        while not should_terminate():
            i = ri(nm1)
            j = ri(nm1)
            if i > j:
                i, j = j, i
            if (i == j) or (i == 0 and j == nm2):
                continue
            useFFA = not useFFA
            if useFFA:
                yd = _my_fea_loop_body(i, j, n, matrix, xd, yd, H)
                if H[yd] <= 1:  # if FEA finds a new solution
                    yc = yd
                    np.copyto(xc, xd)
                register(xd, yd)
            else:
                yc = _my_ea_loop_body(i, j, n, matrix, xc, yc)
                register(xc, yc)

    def __str__(self):
        """
        Get the name of this algorithm.

        This name is then used in the directory path and file name of the
        log files.

        :returns: eafea2_revn
        """
        return "eafea2_revn"
