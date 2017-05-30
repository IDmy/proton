import numpy as np
import matplotlib.pyplot as plt
from proton.estimator import *

class AccuracyHandler(object):
    """AccuracyHandler handles evaluating of LP model using different BED matrixes."""

    # from .optimizer import LPOptimizer

    def __init__(self, actual_BED, gran_range):
        self.actual_BED = actual_BED
        self.gran_range = gran_range

    def get_naive_solution(self, optimizer):
        "Outputs objective value of naive solution using avarage of BED values."
        naive_BED = np.ones(shape=self.actual_BED.shape) * self.actual_BED.mean()
        return self._run_model(optimizer, naive_BED)

    def get_true_solution(self, optimizer):
        "Outputs objective value of optimal solution."
        return self._run_model(optimizer, self.actual_BED)

    def get_predictor_solution(self, predictor, optimizer):
        """Returns a dict of objective values given granularity.
        @param predictor: Must be LinearBEDPredictor (inhereted) object.
        """
        result = {}
        for gran in self.gran_range:
            # What is the optimum found using our estimation?
            estimated = predictor.estimate(granularity=gran)
            result[gran] = self._run_model(optimizer, estimated)
        return result

    def _sum_BED(self, BED, solution):
        """Outputs the total BED of a proposed solution"""
        total = 0
        for i, j in solution.items():
            total += BED[i, j]
        return total

    def _run_model(self, optimizer, BED, capacity=100):
        """
        Run a LP model and get the solution given a BED matrix. Be VERY careful here,
        the ESTIMATED BED should be used to create and solve the model but the
        ACTUAL BED should be used when evaluating the quality of the solution!
        """
        optimizer.build(BED, capacity=capacity)
        solution = optimizer.get_optimum()
        return self._sum_BED(BED, solution)

    def _get_coords(self, estimates):
        """Returns x, y coords given a lits of tuples of coords."""
        coords = sorted(estimates.items())
        x, y = zip(*coords)  # unpack a list of pairs into two tuples
        return x, y

    def get_bound_error(self, lower_bounds, upper_bounds):
        """Computes percentage difference between lower and upper bounds
        @param lower_bounds, upper_bounds: is a dict like {granularity:objective function - BED}
        returns: a dict that looks like {granularity:pertance_difference}
        """
        return {i: (upper_bounds[i] - lower_bounds[i]) * 100 / lower_bounds[i] for i in self.gran_range}

    def draw_evaluation_plot(self, optimizer):
        """Draws a plot of granularity with a relationship to different estimators"""
        linear_predictor = LinearBEDPredictor(self.actual_BED)
        low_bound_lin_internp = self.get_predictor_solution(linear_predictor, optimizer)

        upper_bound_naive_predictor = BEDPredictorUpperBoundNaive(self.actual_BED)
        upper_bound_naive_solution = self.get_predictor_solution(upper_bound_naive_predictor, optimizer)

        upper_bound_correct_predictor = BEDPredictorUpperBoundCorrect(self.actual_BED)
        upper_bound_correct_solution = self.get_predictor_solution(upper_bound_correct_predictor, optimizer)

        avg_BED = self.get_naive_solution(optimizer)
        optimal_BED = self.get_true_solution(optimizer)

        x_li, y_li = self._get_coords(low_bound_lin_internp)
        plt.plot(x_li, y_li, label="interpolated estimation")

        x_wn, y_wn = self._get_coords(upper_bound_naive_solution)
        plt.plot(x_wn, y_wn, label="upper bound naive")

        x_uc, y_uc = self._get_coords(upper_bound_correct_solution)
        plt.plot(x_uc, y_uc, label="upper bound correct")

        gran_range = tuple(self.gran_range)
        # Plot naive estimation results
        plt.plot((1,) + gran_range, [avg_BED] * (len(gran_range) + 1), label="naive estimation", linestyle='dashed')

        # Plot actual optimum
        plt.plot((1,) + gran_range, [optimal_BED] * (len(gran_range) + 1), label="actual BED")

        # Edit plot
        plt.xlim(1, len(gran_range))
        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.xlabel('points observed')
        plt.ylabel('BED')
        plt.xticks(gran_range)
        plt.show()
