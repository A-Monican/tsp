from instance import Instance
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.spaces.permutations import Permutations
from moptipy.utils.path import Path
from tspea1plus1revn import MyEaAlgorithm as earevn
from tspfea1plus1revn import MyFeaAlgorithm as fearevn
from tspeafearevn import MyEafeaAlgorithm as eafearevn
from tspeafea2revn import MyEafea2Algorithm as eafea2revn
from tspsarevn import MySaAlgorithm as sarevn
from tspfsarevn import MyFsaAlgorithm as fsarevn
from tspsafea2revn import MySafea2Algorithm as safea2revn
from tspsafearevn import MySafeaAlgorithm as safearevn
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.tabulate_end_results import tabulate_end_results
from moptipy.utils.html import HTML
from moptipy.utils.latex import LaTeX


import os
import psutil

import sys
sys.path.append(os.path.join(
    os.path.dirname(__file__),
    "moptipy-main"))


ns = lambda prc: False if prc is None else (  # noqa: E731
    "make" in prc.name() or ns(prc.parent()))

# should we show the plots?
SHOW_PLOTS_IN_BROWSER = not ns(psutil.Process(os.getppid()))

problems = [lambda: Instance('a280.tsp'),
            lambda: Instance('berlin52.tsp'),
            lambda: Instance('bier127.tsp'),
            lambda: Instance('eil51.tsp'),
            lambda: Instance('eil76.tsp'),
            lambda: Instance('eil101.tsp'),
            lambda: Instance('kroA100.tsp'),
            lambda: Instance('kroB100.tsp'),
            lambda: Instance('kroC100.tsp'),
            lambda: Instance('kroD100.tsp'),
            lambda: Instance('kroE100.tsp'),
            lambda: Instance('lin105.tsp'),
            lambda: Instance('pr76.tsp'),
            lambda: Instance('pr107.tsp'),
            lambda: Instance('pr124.tsp'),
            lambda: Instance('rat99.tsp'),
            lambda: Instance('st70.tsp'),
            lambda: Instance('ts225.tsp')]  # 18 selected instances


def make_fearevn(problem) -> Execution:
    """
    Create an application of our algorithm to our problem.

    :param problem: the problem (problem)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(fearevn(problem))  # apply our (1+1)FEA with reversing operator
    ex.set_max_fes(10000000000)  # permit 10B FEs
    ex.set_log_improvements(True)  # store all improvements in the log file
    return ex


def make_revn(problem) -> Execution:
    """
    Create an application of our algorithm to our problem.

    :param problem: the problem (problem)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(earevn(problem))  # apply our (1+1)EA with reversing operator
    ex.set_max_fes(10000000000)  # permit 10B FEs
    ex.set_log_improvements(True)  # store all improvements in the log file
    return ex


def make_eafearevn(problem) -> Execution:
    ex = Execution()
    ex.set_solution_space(Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(eafearevn(problem))  # apply our algorithm EAFEA(B)
    ex.set_max_fes(10000000000)  # 10B FEs
    ex.set_log_improvements(True)
    return ex


def make_eafea2revn(problem) -> Execution:
    ex = Execution()
    ex.set_solution_space(Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(eafea2revn(problem))  # apply our algorithm EAFEA(A)
    ex.set_max_fes(10000000000)  # 10B FEs
    ex.set_log_improvements(True)
    return ex


def make_sarevn(problem) -> Execution:
    ex = Execution()
    ex.set_solution_space(Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(sarevn(problem))  # apply our algorithm SA
    ex.set_max_fes(10000000000)  # 10B FEs
    ex.set_log_improvements(True)
    return ex


def make_fsa(problem) -> Execution:
    ex = Execution()
    ex.set_solution_space(Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(fsarevn(problem))  # apply our algorithm FSA
    ex.set_max_fes(10000000000)  # 10B FEs
    ex.set_log_improvements(True)
    return ex


def make_safea(problem) -> Execution:
    """
    Create an application of our algorithm to our problem.

    :param problem: the problem (MySortProblem)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(
        Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    ex.set_algorithm(safearevn(problem))  # apply our algorithm SAFEA(B)
    ex.set_max_fes(10000000000)  # 10B FEs
    # ex.set_max_time_millis(43200000)
    ex.set_log_improvements(True)
    return ex


def make_safea2(problem) -> Execution:
    """
    Create an application of our algorithm to our problem.

    :param problem: the problem (MySortProblem)
    :returns: the execution
    """
    ex = Execution()
    ex.set_solution_space(
        Permutations.standard(problem.city_number))  # we use permutations of [0..n-1]
    ex.set_objective(problem)  # set the objective function
    # print(problem.LB)
    ex.set_algorithm(safea2revn(problem))  # apply our algorithm SAFEA(A)
    ex.set_max_fes(10000000000)  # 10B FEs
    # ex.set_max_time_millis(43200000)
    ex.set_log_improvements(True)
    return ex


td = Path.directory(os.path.dirname(__file__)).resolve_inside("results")
td.ensure_dir_exists()
# The log files will be stored in this target folder
# can be something like this " td = 'D:/pycode/moptipy/mylog' "

run_experiment(
    base_dir=td,  # set the base directory for log files
    instances=problems,  # define the problem instances
    setups=[make_fsa, make_eafea2revn, make_eafearevn, make_fearevn, make_revn, make_safea, make_sarevn, make_safea2],  # setups
    n_runs=1,  # 51 runs to do
    n_threads=1)  # we use only a single thread here


data = []  # we will load the data into this list
EndResult.from_logs(td, data.append)  # load all end results

file = tabulate_end_results(data, dir_name=td)  # create the table
print(f"\nnow presenting markdown data from file '{file}'.\n")
print(file.read_all_str())  # print the result

file = tabulate_end_results(data, dir_name=td,
                            text_format_driver=LaTeX.instance)
print(f"\nnow presenting LaTeX data from file '{file}'.\n")
print(file.read_all_str())  # print the result

file = tabulate_end_results(data, dir_name=td,
                            text_format_driver=HTML.instance)
print(f"\nnow presenting HTML data from file '{file}'.\n")
print(file.read_all_str())  # print the result


EndResult.from_logs(  # parse all log files and print end results
    td, lambda er: print(f"{er.algorithm} on {er.instance}: {er.best_f}"))
