import os
import sys

import numpy as np
import pandas as pd
from moptipy.api.objective import Objective
from moptipy.utils.path import Path


class Instance(Objective):
    def __init__(self, file_name: str) -> None:
        """
        Loading a Euclidean 2d symmetrical tsp instance and initializing some of its parameters
        Please add the length of the optimal solution of this instance to the lbdict before using it
        """
        lbdict = {'st70': 675, 'eil101': 629, 'berlin52': 7542, 'a280': 2579, 'bier127': 118282, 'eil51': 426, 'eil76': 538,
                  'kroA100': 21282, 'kroB100': 22141, 'kroC100': 20749, 'kroD100': 21294, 'kroE100': 22068, 'lin105': 14379,
                  'pr76': 108159, 'pr107': 44303, 'pr124': 59030, 'rat99': 1211, 'ts225': 126643}  # store the the optimal value of these Euc2d instances as the lower bound
        try:
            file_name = file_name  # load the tsp instance
            df = pd.read_csv(
                Path.directory(os.path.dirname(__file__))
                .resolve_inside("tsplib")
                .resolve_inside(file_name), sep=" ", skiprows=6, header=None)
            city = np.array(df[0][0:len(df)-1])
            city_name = city.tolist()
            city_x = np.array(df[1][0:len(df)-1])
            city_y = np.array(df[2][0:len(df)-1])
            city_location = list(zip(city_x, city_y))
            city_number = len(city_name)
        except FileNotFoundError:
            sys.exit("I can not find it...")

        def get_dab(a, b, city_location):  # get the distance between a and b

            def ct_distance(l1):
                """calculate the Euclidean distance between 2 cities in l1, round down"""
                d = np.sqrt((l1[0][0]-l1[1][0]) ** 2 + (l1[0][1] - l1[1][1]) ** 2)
                dis = int(0.5 + d)
                return dis

            a = city_location[a]
            b = city_location[b]
            dab = [a, b]
            return ct_distance(dab)

        cost_tsp = np.zeros((city_number, city_number))
        for i in range(0, city_number):
            for j in range(0, city_number):
                cost_tsp[i][j] = get_dab(i, j, city_location)  # Create a cost matrix for this instance

        """
        In fact, you can compute or generate an upper bound in other ways,
        as long as it is guaranteed to be larger than all feasible solutions in this instance

        """
        index_max = np.argmax(cost_tsp, axis=1)
        max = cost_tsp[range(cost_tsp.shape[0]), index_max]
        upperbound = sum(max)  # The sum of the distance between each city and the city farthest from it
        UB = int(upperbound)+1  # use this distance + 1 as the upperbound of this instance

        self.name = file_name[:-4]
        self.city_number = city_number
        self.cost_tsp = cost_tsp
        self.LB = (lbdict.get(self.name))  # get the lower bound from the lb dict
        self.UB = UB

    def evaluate(self, x) -> int:
        """
        Get the tour distance as the objective value from the solution x.

        x is a list containing all cities and only once
        x is a feasible solution for this tsp instance

        :returns: tour_total_distance
        """
        tour_total_distance = 0
        for i in range(0, self.city_number):
            if i == self.city_number-1:
                a = x[i]
                b = x[0]
            else:
                a = x[i]
                b = x[i+1]
            tour_total_distance = tour_total_distance + self.cost_tsp[a-1][b-1]
        return tour_total_distance

    def lower_bound(self) -> int:
        """
        Get the lower bound of this instance

        Implementing this function is optional, but it can help in two ways:
        First, the optimization processes can be stopped automatically when a
        solution of this quality is reached. Second, the lower bound is also
        checked when the end results of the optimization process are verified.

        :returns: self.LB
        """
        return self.LB

    def upper_bound(self) -> int:
        """
        Get the upper bound of this instance

        Implementing this function is optional, but it can help us to
        know the quality of the solution. The results of the optimization
        process are automatically checked when the value of the solution is
        greater than this upper bound.

        :returns: self.UB
        """
        return self.UB

    def __str__(self) -> str:
        """
        Get the name of this instance

        :returns: f"{self.name}"
        """
        return f"{self.name}"
