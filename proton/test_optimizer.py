import random
import unittest

import numpy as np

from proton.optimizer import LPOptimizer

class ModelingTest(unittest.TestCase):
    def setUp(self):
        """Create a single Proton optimization model."""
        pass

    def mock_BED_data(self):
        BED = np.array([[1, 10, 11], [1, 2, 3], [9, 10, 11]])
        max_fractions = BED.shape[1] - 1
        return BED, max_fractions

    def test_infinite_capacity(self):
        """When capacity is infinite, we expect max amount of fractions for all patients"""
        BED, max_fractions = self.mock_BED_data()
        inf_capacity = 10000
        optimizer = LPOptimizer()
        optimizer.build(BED, capacity=inf_capacity)
        for patient, fractions in optimizer.get_optimum().items():
            self.assertEqual(fractions, max_fractions)

    def test_capacity_constraint(self):
        """An optimal model surely uses as many fractions as it can"""
        BED, max_fractions = self.mock_BED_data()
        num_patients = BED.shape[0]

        # Capacity should be unable to fulfill every patient
        capacity = random.randint(1, max_fractions * num_patients)

        optimizer = LPOptimizer()
        optimizer.build(BED, capacity=capacity)
        solution = optimizer.get_optimum()
        fractions_used = sum(solution.values())
        self.assertEqual(fractions_used, capacity)

    def tearDown(self):
        """Delete all models."""
        pass

if __name__ == "__main__":
    # Run tests
    a = ModelingTest()
    suite = unittest.TestLoader().loadTestsFromModule(a)
    unittest.TextTestRunner().run(suite)