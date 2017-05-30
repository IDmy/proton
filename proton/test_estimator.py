import pandas as pd

from proton.estimator import AccuracyHandler, LinearBEDPredictor, BEDPredictorUpperBoundNaive
from proton.optimizer import LPOptimizer

# ToDo:
# Plotting code should either go to the data exploration notebook or into a dedicated file/module
if __name__ == "__main__":
    # Read data
    data = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
    benefit = data.values

    ah = AccuracyHandler(benefit, range(2, 7))

    optimizer = LPOptimizer()
    ah.draw_evaluation_plot(optimizer)
    lower_bound = ah.get_predictor_solution(LinearBEDPredictor(benefit), optimizer)
    upper_bound = ah.get_predictor_solution(BEDPredictorUpperBoundNaive(benefit), optimizer)

    print("The maximum error between lower and upper bound is ", ah.get_bound_error(lower_bound, upper_bound))
