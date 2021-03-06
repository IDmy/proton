import pandas as pd

from estimator import LinearBEDPredictor, BEDPredictorUpperBoundNaive, BEDPredictorUpperBoundCorrect
from optimizer import LPOptimizer
from accuracy_handler import AccuracyHandler

# Plotting code should either go to the data exploration notebook or into a dedicated file/module
if __name__ == "__main__":
    # Read data
    data = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
    benefit = data.values

    ah = AccuracyHandler(benefit, range(2, 7))

    optimizer = LPOptimizer()
    ah.draw_evaluation_plot(optimizer)
    lower_bound = ah.get_predictor_solution(LinearBEDPredictor(benefit), optimizer)
    upper_bound_naive = ah.get_predictor_solution(BEDPredictorUpperBoundNaive(benefit), optimizer)
    upper_bound_correct = ah.get_predictor_solution(BEDPredictorUpperBoundCorrect(benefit), optimizer)
    #DIMA
    BED_max = BEDPredictorUpperBoundCorrect(benefit).estimate(granularity=4)
    optimizer.build(BED_max, capacity=100)
    solution = optimizer.get_optimum()
    total = 0
    for i, j in solution.items():
        total += BED_max[i, j]
    #!DIMA
    print("The maximum error between lower and naive upper bound is ", ah.get_bound_error(lower_bound, upper_bound_naive))
    print("The maximum error between lower and correct upper bound is ", ah.get_bound_error(lower_bound, upper_bound_correct))