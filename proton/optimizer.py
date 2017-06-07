# GUROBI must be installed and a licence file must be discoverable in one of the default locations
from gurobipy import *

# Get some additional dependencies
import numpy as np
import math

from estimator import LinearBEDPredictor, BEDPredictorUpperBoundCorrect

# Can be overriden as a builder argument
TIME_PER_ACCESS = 5

class ProtonOptimizer(object):
    """
    An abstract class defining a common interface to be used by all our future models.
    Make sure you comply to the API meaning that your concrete class should:
    1) Be constructed by providing the BED values as a 2D np array and the capacity to the build method
    2) It should return its result by implementing the get_optimum() method
    """
    def __init__(self):
        self.BED = None
        self.capacity = 0
        self.optimum = {}

    def build(self, BED, capacity=100, model_name='abstract_optimizer'):
        return self

    def get_type(self):
        return "Abstract"

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
            #print(patient, fractions)
            total += self.BED[patient, fractions]

        print(total)
        return total

    def get_maximum_capacity_needed(self):
        """Returns the maximum (i.e., everyone would get 15 fractions) capacity needed given the BED matrix."""
        num_patients, max_fractions = self.BED.shape
        return num_patients * (max_fractions - 1)

class SmartOptimizer(ProtonOptimizer):
    """"
    This class wraps our concrete optimizers by dynamically selecting the best one for a given job.
    Example use includes its construction, selecting an optimizer given the runtime limitations
    then getting the optimum solution.
    """
    def __init__(self):
        self.optimum = {}
        self.optimizer = None
        self._confidence_rate = None
        super(SmartOptimizer, self).__init__()

    def build(self, BED, capacity=100, max_time=None, model_name='smart_optimizer', time_per_access = TIME_PER_ACCESS, force_linear = False):
        """
        Given the maximum runtime allowed (in minutes), this function will return the
        best underlying optimizer.
        """
        self.BED = BED
        num_patients, max_fractions = BED.shape
        if not max_time:
            # If max_time is not passed we assume no limitation
            max_accesses = num_patients * max_fractions
        else:
            max_accesses = math.floor(max_time / TIME_PER_ACCESS) # number of look-ups

        max_capacity_needed = self.get_maximum_capacity_needed()
        capacity =  max_capacity_needed if max_capacity_needed < capacity else capacity

        heuristic_accesses = num_patients + capacity
        if heuristic_accesses <= max_accesses and not force_linear:
            self.optimizer = HeuristicOptimizer().build(BED, capacity)
        else:
            # If we are short on time we need to use the LP model with an estimated BED matrix.
            granularity = math.floor(max_accesses / num_patients)
            estimated_BED = LinearBEDPredictor(BED).estimate(granularity)
            self.optimizer = LPOptimizer().build(estimated_BED, capacity)
            self._compute_confidence(granularity, BED, capacity)
            self._compute_calculation_time(granularity, BED)
        return self

    def get_confidence_rate(self):
        if self._confidence_rate is not None:
            return self._confidence_rate
        return 0

    def get_calculation_time(self):
        if self._calculation_time is not None:
            return self._calculation_time
        return 0

    def _compute_confidence(self, granularity, BED, capacity):
        BED_max = BEDPredictorUpperBoundCorrect(BED).estimate(granularity=granularity)
        upper_bound_optimizer = LPOptimizer().build(BED_max, capacity)
        upper_bound_optimizer.build(BED_max, capacity=capacity)
        print("test")
        upper_bound_obj = upper_bound_optimizer.get_total_BED()
        print(upper_bound_obj, "upper")
        lower_bound_obj = self.get_total_BED()
        print(lower_bound_obj, "lower")
        confidence_rate = (upper_bound_obj - lower_bound_obj) / lower_bound_obj * 100
        print(confidence_rate)
        self._confidence_rate = confidence_rate

    def _compute_calculation_time(self, granularity, BED):
        NUM_PATIENTS = BED.shape[0]
        print ('granularity: ', granularity)
        calculation_time = granularity*NUM_PATIENTS*5
        self._calculation_time = calculation_time

    def get_optimum(self):
        if not self.optimizer:
            raise ValueError("no optimizer has been selected")
        return self.optimizer.get_optimum()

    def get_type(self):
        return self.optimizer.get_type()


class LPOptimizer(ProtonOptimizer):
    """Concrete linear implementation of the ProtonOptimizer interface"""

    def __init__(self):
        self.m = None
        self.patients = None
        self.fractions = None
        self.x = None
        super(LPOptimizer, self).__init__()

    def get_type(self):
        return "Linear"

    def build(self, BED, capacity=100, model_name='linear_optimizer'):
        self.m = Model(model_name)
        self.m.reset()
        self.BED = BED
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
        return self

    def _solve(self, debug=False):
        # Set objective
        self.m.setObjective(quicksum(
            self.x[i, j] * self.BED[i, j] for i in self.patients for j in self.fractions),
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
        self.num_accesses = 0
        super(HeuristicOptimizer, self).__init__()

    def build(self, BED, capacity=100, model_name='heuristic_optimizer'):
        self.BED = BED
        max_capacity_needed = self.get_maximum_capacity_needed()
        capacity = max_capacity_needed if max_capacity_needed < capacity else capacity
        self.capacity = capacity
        return self

    def get_type(self):
        return "Heuristic"

    def get_optimum(self):
        self.num_accesses = 0
        num_patients = self.BED.shape[0]
        max_fractions_per_patient = self.BED.shape[1] - 1

        # Initialize
        state = [0] * num_patients
        benefit = self.BED[:, 1] - self.BED[:, 0]
        self.num_accesses += num_patients

        for _ in range(self.capacity):
            patient = np.argmax(benefit)
            if state[patient] == max_fractions_per_patient: #this occurs when data are non-concave (benefit[patient] becomes negative)
                break

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
