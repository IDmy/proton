import unittest
from proton.estimator import BEDPredictorUpperBoundCorrect
import numpy as np

class BEDPredictorUpperBoundCorrectTest(unittest.TestCase):

    def test_case_BED_linear(self):
        """Testing BED approximation when BED values are linear."""
        num_fractions = 16
        BED = np.array([range(num_fractions + 1)])
        x = BEDPredictorUpperBoundCorrect(BED)
        estimated = x.estimate(granularity=4)
        self.assertEqual(np.all(np.equal(estimated[0], range(num_fractions + 1))), True)  # the second patient

    def test_case_BED_concave(self):
        """Testing BED approximation when BED values are arbitrary convex."""
        BED = np.array([[0, 20, 38, 54, 68, 80, 90, 98, 104, 108, 110, 111, 111, 111, 111, 111]])
        x = BEDPredictorUpperBoundCorrect(BED)
        estimated = x.estimate(granularity=4) # accessed values are 0, 80, 110, 111 and indexes 0, 5, 10, 15
        slope1_2 = (110 - 80) / 5
        slope0_1 = (80 - 0) / 5
        slope2_3 = (111 - 110) / 5
        ground_truth = [0, 80 - 4 * slope1_2, 80 - 3 * slope1_2, 80 - 2 * slope1_2, 80 - slope1_2, 80, 80 + slope0_1,
                      110 - 3 * slope2_3, 110 - 2 * slope2_3, 110 - slope2_3, 110, 111, 111, 111, 111, 111]
        #the estimates and grouth_truth need to be rounded due to floating point issues
        estimates =  np.around(estimated[0], decimals = 2)
        ground_truth = np.around(ground_truth, decimals = 2)
        self.assertEqual(np.all(np.equal(estimates, ground_truth)), True)

if __name__ == "__main__":
    # Run tests
    a = BEDPredictorUpperBoundCorrectTest()
    suite = unittest.TestLoader().loadTestsFromModule(a)
    unittest.TextTestRunner().run(suite)