import random
import unittest

import numpy as np
import pandas as pd

from optimizer import LPOptimizer, HeuristicOptimizer, SmartOptimizer


class ModelingTest(unittest.TestCase):
    def setUp(self):
        pass

    def mock_BED_data(self):
        BED = np.array([[1, 10, 11], [1, 2, 3], [9, 10, 11]])
        max_fractions = BED.shape[1] - 1
        return BED, max_fractions

    def sample_BED_data(self):
        # Sample dataset read from file featuring 17 patients who can receive a maximum of 15 fractions
        data = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
        BED = data.values
        max_fractions = BED.shape[1] - 1
        return BED, max_fractions

    def infinite_capacity_test(self,optimizer):
        """When capacity is infinite, we expect max amount of fractions for all patients"""
        BED, max_fractions = self.mock_BED_data()
        inf_capacity = BED.shape[0] * max_fractions
        optimizer.build(BED, capacity=inf_capacity)
        for patient, fractions in optimizer.get_optimum().items():
            self.assertEqual(fractions, max_fractions)

    def finite_capacity_test(self, optimizer):
        """An optimal model surely uses as many fractions as it can"""
        BED, max_fractions = self.mock_BED_data()
        num_patients = BED.shape[0]

        # Capacity should be unable to fulfill every patient
        capacity = random.randint(1, max_fractions * num_patients)

        optimizer.build(BED, capacity=capacity)
        solution = optimizer.get_optimum()
        fractions_used = sum(solution.values())
        self.assertEqual(fractions_used, capacity)

    def test_LP_infinite(self):
        optimizer = LPOptimizer()
        self.infinite_capacity_test(optimizer)

    def test_heuristic_infinite(self):
        optimizer = HeuristicOptimizer()
        self.infinite_capacity_test(optimizer)

    def test_LP_capacity(self):
        optimizer = LPOptimizer()
        self.finite_capacity_test(optimizer)

    def test_heuristic_capacity(self):
        optimizer = HeuristicOptimizer()
        self.finite_capacity_test(optimizer)

    def test_accesses(self):
        BED, max_fractions = self.mock_BED_data()
        capacity = 4
        optimizer = HeuristicOptimizer().build(BED, capacity = capacity)
        num_patients = BED.shape[0]
        self.assertEqual(optimizer.get_accesses(), capacity + num_patients)

    def test_smart_optimizer_selection(self):
        BED, _ = self.sample_BED_data()
        opt = SmartOptimizer().build(BED, capacity = 100, max_time = 585)
        self.assertEqual(opt.get_type(), "Heuristic", msg="Given enough time heuristic should be chosen")
        opt = SmartOptimizer().build(BED, capacity = 100, max_time = 300)
        self.assertEqual(opt.get_type(), "Linear", msg="With limited time LP with estimation should be chosen")

    def test_optimizers_agree(self):
        # Under the concavity assumption both optimizers should produce the global optimum
        BED, max_fractions = self.mock_BED_data()
        lp = LPOptimizer()
        heur = HeuristicOptimizer()

        lp.build(BED, capacity = 5)
        heur.build(BED, capacity = 5)
        self.assertEqual(lp.get_optimum(), heur.get_optimum())


    def tearDown(self):
        """Delete all models."""
        pass

if __name__ == "__main__":
    # Run tests
    a = ModelingTest()
    suite = unittest.TestLoader().loadTestsFromModule(a)
    unittest.TextTestRunner().run(suite)