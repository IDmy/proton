# GUROBI must be installed and a licence file must be discoverable in one of the default locations
from gurobipy import *

# Get some additional dependencies
import pandas as pd
import numpy as np
import time


class ProtonOptimizer(object):
    """
    An abstract class defining a common interface to be used by all our future models.
    Make sure you comply to the API meaning that your concrete class should:
    1) Be constructed by providing the BED values as a 2D np array and the capacity to the build method
    2) It should return its result by implementing the get_optimum() method
    """

    def build(self, BED, capacity=100, model_name='abstract_optimizer'):
        raise NotImplementedError("You cannot build an abstract model")

    def get_optimum(self):
        """Returns a dictionary (int -> int) from patient ID to fractions"""
        return {}

    def pretty_print(self):
        solution = self.get_optimum()
        for patient, fractions in solution.items():
            print(("Patient " + str(patient) + " should receive " + str(fractions) + " fractions"))

    def get_total_BED(self):
        total = 0
        for patient, fractions in self.get_optimum().items():
            total += self.BED[patient, fractions]
        return total

class LPOptimizer(ProtonOptimizer):
    """Concrete linear implementation of the ProtonOptimizer interface"""

    def __init__(self):
        self.optimum = {}
        self.m = None
        self._BED = None
        self.patients = None
        self.fractions = None
        self.x = None

    def build(self, BED, capacity=100, model_name='linear_optimizer'):
        self.m = Model(model_name)
        self.m.reset()
        self._BED = BED
        num_patients, max_fractions_per_patient = BED.shape
        self.patients = [i for i in range(num_patients)]
        self.fractions = [j for j in range(max_fractions_per_patient)]

        # Set binary decision variables
        self.x = self.m.addVars(num_patients, max_fractions_per_patient, vtype=GRB.BINARY)

        # Only one choice of fractions per patient is valid
        self.m.addConstrs(quicksum(self.x[i, j] for j in self.fractions) == 1 for i in self.patients)

        # We can only perform so many proton therapies per week
        self.m.addConstr(quicksum(
            quicksum(self.x[i, j] * self.fractions[j] for j in self.fractions)
            for i in self.patients) <= capacity)
        self.m.update()

    def _solve(self, debug=False):
        # Set objective
        self.m.setObjective(quicksum(
            self.x[i, j] * self._BED[i, j] for i in self.patients for j in self.fractions),
            GRB.MAXIMIZE)

        self.m.setParam('OutputFlag', debug)
        self.m.update()
        self.m.optimize()
        if self.m.status == GRB.Status.OPTIMAL:
            solution = self.m.getAttr("x", self.x)
            for i in self.patients:
                for j in self.fractions:
                    if (solution[i, j] == 1):
                        self.optimum[i] = j
                        break
        else:
            print("Infeasible model")

    def get_optimum(self):
        self._solve()
        return self.optimum


class HeuristicOptimizer(ProtonOptimizer):
    """
    Concrete heuristic implementation of the ProtonOptimizer interface
    This will produce the globally optimum solution assuming concavity in the BED matrix
    which may or may not be true in the real case
    """
    def __init__(self):
        self.optimum = {}
        self.BED = None

    def build(self, BED, capacity=100, model_name='heuristic_optimizer'):
        self.BED = BED
        self.capacity = capacity
        self.num_accesses = 0

    def get_optimum(self):
        num_patients = self.BED.shape[0]
        max_fractions_per_patient = self.BED.shape[1] - 1

        # Initialize
        state = [0] * num_patients
        benefit = self.BED[:, 1] - self.BED[:, 0]
        self.num_accesses += num_patients

        for i in range(self.capacity):
            patient = np.argmax(benefit)
            value = benefit[patient]
            state[patient] = state[patient] + 1
            if state[patient] == max_fractions_per_patient:
                benefit[patient] = 0
            else:
                benefit[patient] = self.BED[patient, state[patient] + 1] - self.BED[patient, state[patient]]
            self.num_accesses += 1

        return dict(zip(range(num_patients), state))

    def get_accesses(self):
        if not self.num_accesses:
            self.get_optimum()
        return self.num_accesses
