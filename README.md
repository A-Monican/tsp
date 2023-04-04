# Experiments with Frequency Fitness Assignment based hybrids on the Traveling Salesperson Problem


## 1. Introduction

The implementation and experimental results of 8 different algorithms to solve the `EUC_2D` Traveling Salesperson Problem (TSP) instances from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

A TSP is defined by a fully-connected weighted graph of `n` cities.
The goal is to find the overall shortest tour that visits each cities exactly once and returns to its starting point.
The TSP is NP-hard.
We consider 18 symmetric Euclidean instances from the well-known [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

Solutions in our work are stored in the path representation, where such a tour is encoded as a permutation `x` of the numbers `1` to `n`, each identifying a city.
If a city appears at index `j` in the permutation `x`, then it will be the `j`<sup>th</sup> city to be visited.
This means that a tour `x` will pass the following edges:
`(x[1], x[2])`, `(x[2], x[3])`, `(x[3], x[4])`, &hellip; `(x[n-1], x[n])`, `(x[n], x[1])`.

The (1+1)&nbsp;EA is the most basic evolutionary algorithm and also be considered as a randomized local search.
It starts with one random solution/permutation `xc` and computes its length `yc=f(xc)`.
In each iteration, it applies a unary search operator `op` to obtain a new tour `xn=op(xc)` and computes its length `yn=f(xn)`.
If `yn<=yc`, then it will accept the new tour and set `xn=xn` and `yc=yn`.

FFA is a fitness assignment process that takes place before this last step in the EA.
We integrate FFA into the (1+1)&nbsp;EA and obtain the (1+1)&nbsp;FEA.
This algorithm uses an additional table `H` which counts, for any tour length `y`, how often it has been seen during the search so far.
After the new tour `xn` is created and its objective value `yn` is computed, the (1+1)&nbsp;FEA sets `H[yc] = H[yc] + 1` and `H[yn] = H[yn] + 1`.
It will accept `xn` if and only if `H[yn] <= H[yc]` and, only in this case, set `xn=xn` and `yc=yn`.

SA is the classical simulated annealing algorithm, in our experiment, it will accept the new solution `xn` with probability `P`, although the better solution accepts probability `P` is 1, it also accepts the worse solution.
This algorithm has a temperature cooling schedule.
As the running time goes on, the temperature decreases and the probability `P` of accepting the worse solution decreases.

EAFEA(A) is a hybrid which alternates between the EA and the FEA and copies a solution from the FEA to the EA if it has an entirely new objective value.
SAFEA(B) is a hybrid which alternates between the SA and the FEA and copies a solution from the FEA to the SA part if it has a better objective value.

We apply all algorithms with the same unary operator.

`reverse` reverses a randomly chosen subsequence of the tour.


## 2. Directory Structure

This archive contains the following directories:

- `results_and_evaluation` contain the results of 8 experiments as well as their evaluation.
    - `results` is the directory with the log files
    - `evaluation` is a folder with the extracted evaluation and figures
	- `evaluation_edited` is a folder with the edited evaluation and figures
    - `evaluator` is a folder with a Python script `main.py` that generates all the files in `evaluation` from the data it finds in `results`.
      It requires the [`moptipy`](https://thomasweise.github.io/moptipy) package being installed for running.
- `source` contains the Python source codes needed to run the `results` experiment.
  - `moptipy-main` is a local copy of the [`moptipy`](https://thomasweise.github.io/moptipy) package used for our experiment.
  - `tsplib` contains the [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) data.
     This includes the instances used in our experiments as files in text format with suffix `.tsp`.
     If an optimal tour is given, it is stored in a text format file with suffix `.opt.tour` and name prefix identical to the instance file.
     In other words, the file `eil51.tsp` contains the TSP instance `eil51` and the file  `eil51.opt.tour` contains the corresponding optimal tour.
     Both the TSP instances and optimal tours can be downloaded from <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/>.
     We also include the documentation of TSPLIB in file [`tsp95.pdf`](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf) documenting them.
     We further include the [TSPLIB FAQ](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html) both as HTML and PDF file (`tsplib_faq.html` and `tsplib_faq.pdf`) and the [list of known optimal tour lengths](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html) as HTML and PDF file (`optimal_tour_lengths_of_symmetric_tsps.html`, `optimal_tour_lengths_of_symmetric_tsps.pdf`).
     Notice that, while the TSP instances we used are Euclidean, all distances are converted to integers as prescribed by the [documentation](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).


## 3. What algorithms are included in this experiment?

We implement 8 different algorithms with EA and SA as the main components.
We use the common path representation for the TSP with `n` cities, which encodes each solution as a permutation of the numbers `1..n`.
The value `i` at position `j`, i.e., `x[j] = i`, in such a permutation indicates that the `j`<sup>th</sup> city to be visited by `i`.
We choose one unary search operators, namely `reverse`, which reverses a subsequence of the tour.
- EA part
	- EA, the simple (1+1) EA algorithm
	- FEA, the (1+1)&nbsp;EA with Frequency Fitness Assignment (FFA)
	- EAFEA&nbsp;(A), which alternates between the EA and the FEA and copies a solution from the FEA to the EA if it has an entirely new objective value
	- EAFEA&nbsp;(B), which alternates between the EA and the FEA and copies a solution from the FEA to the EA if it has a better objective value

- SA part
	- SA, the classical simulated annealing algorithm
	- FSA, SA with Frequency Fitness Assignment (FFA)
	- SAFEA&nbsp;(A), which alternates between the SA and the FEA and copies a solution from the FEA to the SA if it has an entirely new objective value
	- SAFEA&nbsp;(B), which alternates between the SA and the FEA and copies a solution from the FEA to the SA if it has a better objective value


## 4. How to Run the Experiment


First, you must make sure to have all the dependencies installed that this program requires.
You can do this by executing the following command in the terminal:

```
pip install matplotlib numba numpy pandas psutil scikit-learn
```

Now enter the `source` directory, i.e., the directory containing the `run.py` file, in your terminal.
Depending on your system configuration and whether you run Windows or Linux, you can start the program with *one* of the commands below.
(If running the first command returns with an error, just try the next one in the list.)

- `python3 -m run`
- `python -m run`
- `python run.py`
- `python3 run.py`

Then the experiment will run.
It will automatically create a sub-folder `results` in `source` and place all log files that are generated into it.
Be careful:
The experiment will take a long time.
However, if you have multiple CPUs, you can simply start several instances of this program in independent terminals.
Each instance will then conduct different runs.
This also works if this folder is shared over the network, in which case you can run multiple processes on multiple PCs.

Side note:
This experiment uses the [`moptipy`](https://thomasweise.github.io/moptipy) package for implementing its algorithms, running the experiments, and gathering their results.
If you want to install `moptipy` on your system instead of using the version supplied here, you can install it via `pip install moptipy`.


## 5. Literature


- Frequency Fitness Assignment (FFA):
  1. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness Assignment: Making Optimization Algorithms Invariant under Bijective Transformations of the Objective Function Value. *IEEE Transactions on Evolutionary Computation* 25(2):307–319. April 2021. Preprint available at [arXiv:2001.01416v5](http://arxiv.org/abs/2001.01416) [cs.NE] 15&nbsp;Oct&nbsp;2020. doi:[10.1109/TEVC.2020.3032090](http://dx.doi.org/10.1109/TEVC.2020.3032090). Experimental results and source code are available at doi:[10.5281/zenodo.3899474](http://doi.org/10.5281/zenodo.3899474).
  2. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency Fitness Assignment: Optimization without Bias for Good Solutions can be Efficient. [arXiv:2112.00229v4](https://arxiv.org/abs/2112.00229v4) [cs.NE] 25&nbsp;May&nbsp;2022.
  3. Thomas Weise, Mingxu Wan, Ke Tang, Pu Wang, Alexandre Devert, and Xin Yao. Frequency Fitness Assignment. *IEEE Transactions on Evolutionary Computation (IEEE-EC)* 18(2):226-243, April&nbsp;2014. doi:[10.1109/TEVC.2013.2251885](http://dx.doi.org/10.1109/TEVC.2013.2251885).
  4. Thomas Weise, Xinlu Li, Yan Chen, and Zhize Wu. Solving Job Shop Scheduling Problems Without Using a Bias for Good Solutions. In *Genetic and Evolutionary Computation Conference Companion (GECCO'21 Companion),* July 10-14, 2021, Lille, France. ACM, New York, NY, USA. ISBN&nbsp;978-1-4503-8351-6. doi:[10.1145/3449726.3463124](http://doi.org/10.1145/3449726.3463124).
  5. Thomas Weise, Yan Chen, Xinlu Li, and Zhize Wu. Selecting a diverse set of benchmark instances from a tunable model problem for black-box discrete optimization algorithms. *Applied Soft Computing Journal (ASOC)*, 92:106269, June&nbsp;2020. doi:[10.1016/j.asoc.2020.106269](http://dx.doi.org/10.1016/j.asoc.2020.106269).
  6. Thomas Weise, Mingxu Wan, Ke Tang, and Xin Yao. Evolving Exact Integer Algorithms with Genetic Programming. In *Proceedings of the IEEE Congress on Evolutionary Computation (CEC'14), Proceedings of the 2014 World Congress on Computational Intelligence (WCCI'14)*, pages&nbsp;1816-1823, Beijing, China, July&nbsp;6-11, 2014. Los Alamitos, CA, USA: IEEE Computer Society Press. ISBN:&nbsp;978-1-4799-1488-3. doi:[10.1109/CEC.2014.6900292](http://dx.doi.org/10.1109/CEC.2014.6900292).
- Traveling Salesperson Problem (TSP):
  1. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, I. Inza, and S. Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A Review of Representations and Operators. *Artificial Intelligence Review,* 13(2):129–170, April 1999. Kluwer Academic Publishers, The Netherlands. doi:[10.1023/A:1006529012972](https://doi.org/10.1023/A:1006529012972).
  2. Gerhard Reinelt. TSPLIB &mdash; A Traveling Salesman Problem Library. *ORSA Journal on Computing* 3(4):376-384. 1991. <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>.
  3. Gerhard Reinelt. TSPLIB95. 1995. Heidelberg, Germany: Universität Heidelberg, Institut für Angewandte Mathematik. <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf>.
  4. Thomas Weise, Raymond Chiong, Ke Tang, Jörg Lässig, Shigeyoshi Tsutsui, Wenxiang Chen, Zbigniew Michalewicz, and Xin Yao. Benchmarking Optimization Algorithms: An Open Source Framework for the Traveling Salesman Problem. *IEEE Computational Intelligence Magazine (CIM)* 9(3):40-52, August&nbsp;2014. doi:[10.1109/MCI.2014.2326101](http://dx.doi.org/10.1109/MCI.2014.2326101).
  5. Eugene Leighton Lawler, Jan Karel Lenstra, Alexander Hendrik George Rinnooy Kan, and David B. Shmoys. *The Traveling Salesman Problem: A Guided Tour of Combinatorial Optimization.* Wiley Interscience. 1985.
  6. David Lee Applegate, Robert E. Bixby, Vasek Chvatal, and William John Cook. *The Traveling Salesman Problem: A Computational Study.* Princeton University Press. 2007.
  7. Gregory Z. Gutin and Abraham P. Punnen, editors. *The Traveling Salesman Problem and its Variations.* Volume 12 of Combinatorial Optimization. Kluwer Academic Publishers. 2002. doi:[10.1007/b101971](https://dx.doi.org/10.1007/b101971).
- Software:
  1. The Metaheuristic Optimization in Python Package [`moptipy`](https://thomasweise.github.io/moptipy)


## 6. License

The files in this repository are under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode), with the exception of the files of [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) in directory `source/tsplib`, which are under copyright of their respective owner (we believe that they are in the public domain, as they are provided by many sources, included in many software packages under various open source licenses, and on many websites).
The license is contained as file `LICENSE` in this archive.


## 7. Contact

If you have any questions or suggestions, please contact 

Mr. Tianyu LIANG (梁天宇) of the 
Institute of Applied Optimization (应用优化研究所, [IAO](http://iao.hfuu.edu.cn)) of the
School of Artificial Intelligence and Big Data ([人工智能与大数据学院](http://www.hfuu.edu.cn/aibd/)) at
[Hefei University](http://www.hfuu.edu.cn/english/) ([合肥学院](http://www.hfuu.edu.cn/)) in
Hefei, Anhui, China (中国安徽省合肥市) via
email to [liangty@stu.hfuu.edu.cn](mailto:liangty@stu.hfuu.edu.cn).