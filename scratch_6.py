import math

from scipy.spatial import KDTree
from collections import defaultdict
from math import sqrt
from operator import itemgetter
from collections import OrderedDict
import collections

import scipy.spatial
import numpy
from ortools.linear_solver import pywraplp


def distance(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def min_edge_cover(points):
    # Enumerate the candidate edges.
    candidate_edges = set()
    tree = scipy.spatial.KDTree(points)
    min_distances = numpy.ndarray(len(points))
    for i, p in enumerate(points):
        print(i)
        distances, indexes = tree.query(p, k=2)
        # Ignore p itself.
        d, j = (
            (distances[1], indexes[1])
            if indexes[0] == i
            else (distances[0], indexes[0])
        )
        candidate_edges.add((min(i, j), max(i, j)))
        min_distances[i] = distances[1]
    brojac_dis = 0
    for i, p in enumerate(points):
        print(i)
        # An edge is profitable only if it's shorter than the sum of the
        # distance from each of its endpoints to that endpoint's nearest
        # neighbor.
        indexes = tree.query_ball_point(p, 3 * min_distances[i])
        for j in indexes:
            if i == j:
                continue
            discount = (
                min_distances[i]**2 + min_distances[j]**2
            ) - (scipy.spatial.distance.euclidean(points[i], points[j]))**2
            if discount > 0:
                brojac_dis+=1
                candidate_edges.add((min(i, j), max(i, j)))
    candidate_edges = sorted(candidate_edges)
    print(brojac_dis)
    # Formulate and solve a mixed integer program to find the minimum distance
    # edge cover. There's a way to do this with general weighted matching, but
    # OR-Tools doesn't expose that library yet.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    objective = 0
    edge_variables = []
    coverage = collections.defaultdict(lambda: 0)
    for i, j in candidate_edges:
        x = solver.BoolVar("x{}_{}".format(i, j))
        objective += (scipy.spatial.distance.euclidean(points[i], points[j]))**2 * x
        xs = points[i][0]+points[j][0]
        ys = points[i][1]+points[j][1]
        rs = distance(points[i][0],points[i][1],points[j][0],points[j][1])/2
        indexes = tree.query_ball_point([xs,ys],rs)
        for index in indexes:
            if index!=i and index!=j:
                coverage[index] += x
        coverage[i] += x
        coverage[j] += x
        edge_variables.append(x)
    solver.Minimize(objective)
    for c in coverage.values():
        solver.Add(c >= 1)
    solver.EnableOutput()
    assert solver.Solve() == pywraplp.Solver.OPTIMAL
    return {e for (e, x) in zip(candidate_edges, edge_variables) if x.solution_value()}


def random_point():
    return complex(random(), random())


def test(points, graphics=False):
    cover = min_edge_cover(points)
    rezultat = 0
    f = open("sunc2.txt", "w")
    for point in cover:
        index1 = point[0]
        index2 = point[1]
        x1 = arr_ulaz[index1][0]
        y1 = arr_ulaz[index1][1]
        x2 = arr_ulaz[index2][0]
        y2 = arr_ulaz[index2][1]
        rs = distance(x1,y1,x2,y2)
        rs /=2
        xs = (x1 + x2)/2
        ys = (y1 + y2)/2
        rezultat += rs**2*3.14159
        tmp = ""
        tmp += str(xs) + ' ' + str(ys) + ' ' + str(rs) + ' ' + str(2) + ' ' + str(index1+1) + ' ' + str(index2+1) + '\n'
        f.write(tmp)
    print(rezultat)
    return


arr_ulaz = numpy.genfromtxt("ulaz.csv", delimiter=",")
arr_ulaz = arr_ulaz[1:]
test(arr_ulaz, graphics=False)


