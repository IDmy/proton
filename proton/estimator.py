import pandas as pd
import numpy as np
import time

from itertools import chain
import matplotlib.pyplot as plt


class LinearBEDPredictor(object):
    """
    A base implementation of BED matrix estimation. Given the real BED matrix and
    a time cost per access, the predictor can output an estimation of the BED matrix along
    with the time needed to produce it.

    EXTENSIONS: If you want to try another predictor (f.e non linear interpolation or series prediction)
    all you have to do is inherit this base and override the estimate() function. Make sure you
    obey its contract: it should return the predicted BED matrix as a 2D np.array. Everything else will
    just work.
    """

    def __init__(self, BED):
        self._BED = BED
        self.num_patients, self.max_fractions_per_patient = BED.shape
        self.access_counter = 0
        self._accessed = set()

    def _getBED(self, i, j):
        """
        A class method simulating the actual generation of BED values.
        If the requested value has been already computed it returns without delay, else it
        delays for the time needed to compute it.
        """
        if (i, j) not in self._accessed:
            self.access_counter += 1
            self._accessed.add((i, j))
            # replace this by actual time cost, i.e 5*60 seconds to simulate real world application.
            time.sleep(0)

        return self._BED[i, j]

    def root_mean_square_error(self, granularity):
        """
        Computes and returns the mean squared error between the actual and the estimated BED
        matrix.
        """
        estimated = self.estimate(granularity)
        difference = estimated - self._BED
        return np.sqrt((difference ** 2).mean(axis=None))

    def estimate(self, granularity):
        """
        Check some points in regural intervals and linearly interpolate the rest.

        @param granurality: number of actual measurements to check
        @return estimate, cost: estimated BED matrix, time cost in seconds to compute it
        """
        if (granularity <= 1 or granularity > (self.max_fractions_per_patient - 1) / 2):
            print("Granularity should be between 2 and <num_columns/2>!")
            return None

        self.access_counter = 0
        self._accessed = set()

        div = (self.max_fractions_per_patient - 1) / (granularity - 1)

        # Retrieve actual values with the simulated getter - This is time consuming.
        actual_indeces = [int(i * div) for i in range(granularity)]

        actual_values = [self._getBED(i, j) for i in range(self.num_patients) for j in actual_indeces]

        estimate = np.empty(shape=(self.num_patients, self.max_fractions_per_patient)) * np.nan
        # Init with actual measurements
        estimate[:, actual_indeces] = np.reshape(actual_values, (self.num_patients, len(actual_indeces)))
        # Interpolate the rest linearly
        estimate = pd.DataFrame(estimate).interpolate(axis=1).values

        return estimate


class BEDPredictorUpperBoundNaive(LinearBEDPredictor):
    """
    This is a class that helps to estimate upper bound of BED using a naive method that assumes
    that the all point between x[i] and x[j] will be of a value of x[j].
    """

    def get_interp_gap(self, accessed_inds, start, end):
        """Returns how many elements will have the same value betwen two interpolated points.
         E.g. if x=2 and x=4 are interpolated, the function return 4-2 = 2.
        """
        if start > end:
            raise ValueError("Start index cannot be bigger than end index.")

        if start < 0 or len(accessed_inds) < end + 1:  # if the indices are out of range
            return 1

        return accessed_inds[start] - accessed_inds[end]

    def _get_first_n_estimates(self, accessed_inds):
        """Computes estimates up to the last looked-up index in BED.
        E.g. if accessed indeces [0, 8, 15], value 16 is not estimated in this step.
        @param accessed_inds: list looked-up indexes in BED
        returns: np.array of a shape of (number of patients, first n values)
        """
        # Reverse the indexes, so it is easy to insert max bounds for each value.
        # Max bound of x[i, j] = BED[i, next access value]
        accessed_inds_rev = list(reversed(accessed_inds))

        first_n_estimates = [[self._getBED(i, j)] * self.get_interp_gap(accessed_inds_rev, ind, ind + 1) \
                             for i in range(self.num_patients) for ind, j in enumerate(accessed_inds_rev)]
        first_n_estimates = list(chain.from_iterable(first_n_estimates))  # make the list 1d
        first_n_estimates = np.reshape(first_n_estimates,
                                       (self.num_patients, accessed_inds_rev[0] + 1))  # reshape into matrix
        first_n_estimates = np.apply_along_axis(lambda row: row[::-1], 1, first_n_estimates)  # reversing order
        return first_n_estimates

    def _get_last_n_estimates(self, accessed_inds):
        """Computes estimates after the last looked-up value in BED.
        E.g. if accessed indeces [0, 8, 15], value 16 is estimated in this step.
        The computation is done using assumption of concativity; the growth of last values will
        not exceed the growth of first_n_estimates[-1] - first_n_estimates[-2].

        @param accessed_inds: list looked-up indexes in BED
        returns: np.array of a shape of (number of patients, last n values)
        """
        last_n_est_count = self.max_fractions_per_patient - accessed_inds[-1] - 1  # how many values to ustimate
        last_n_estimates = [self._getBED(i, accessed_inds[-1]) + (self._getBED(i, accessed_inds[-1]) - \
                                                                  self._getBED(i, accessed_inds[-2])) * (j + 1) for i in
                            range(self.num_patients) \
                            for j in range(last_n_est_count)]
        last_n_estimates = np.reshape(last_n_estimates, (self.num_patients, last_n_est_count))  # reshape into matrix
        return last_n_estimates

    def estimate(self, granularity):
        """
        Check some points in regural intervals and linearly interpolate the rest.

        @param granurality: number of actual measurements to check
        @return estimate, cost: estimated BED matrix, time cost in seconds to compute it
        """
        if (granularity <= 1 or granularity > (self.max_fractions_per_patient - 1) / 2):
            print("Granularity should be between 2 and <num_columns/2>!")
            return None

        self.access_counter = 0
        self._accessed = set()

        interp_step = (self.max_fractions_per_patient - 1) / (granularity - 1)  # take out -1

        # Which indices will and will not be access in BED look-ups
        accessed_inds = [int(i * interp_step) for i in range(granularity)]

        if 0 not in accessed_inds:
            raise ValueError(
                "The values to bed access in BED does not contain index 0. Estimation of BED will not work correctly.")

        first_n_estimates = self._get_first_n_estimates(accessed_inds)
        last_n_estimates = self._get_last_n_estimates(accessed_inds)

        BED_estimates = np.c_[first_n_estimates, last_n_estimates]  # column-concatente first and last values
        return BED_estimates



#from sklearn.metrics import mean_squared_error
class BED_estimate(LinearBEDPredictor):

    def besenham(self, m, n):
        '''
        Choose m evenly spaced elements from a sequence of length n
        '''
        idx = [i * n // m + n // (2 * m) for i in range(m)]
        return idx

    def estimate_vector(self, input_vector, threshold, length_output_vector):
        '''
        An alternative estimation method of the BED matrix.

        Parameters
        -----------
        input_vector: list
                A n-length list
        threshold : int
                The threshold value
        output_vector: int
                The desired m-length estimated BED vector
        Returns
        -----------
        list
            The estimated m-length BED vector

        Raises
        -----------
        ValueError
            If input_vector>output_vector

        '''
        length_of_input = len(input_vector)
        # Raise error in case that the length of the output vector is smaller than the input vector
        if length_output_vector < length_of_input:
            raise ValueError("Granularity of output should be bigger or equal than input")

        # The output vector
        output = np.zeros(length_output_vector)
        # In order to keep the same indexes, the 'new' input vector is equally size as the output vector
        extend_input = np.zeros_like(output)
        extend_input[0], extend_input[-1] = input_vector[0], input_vector[-1]
        # Bresenham's line algorithm
        # Note: I already allocate two points of the input vector
        m = length_of_input - 2
        n = length_output_vector - 1
        for (x, y) in zip(self.besenham(m, n), input_vector[1:-1]):
            extend_input[x] = y
        idxs = list(np.nonzero(extend_input)[0])
        for idx_1, idx_2 in zip(idxs[0:-1], idxs[1:]):
            temp_1 = extend_input[idx_1]
            temp_2 = extend_input[idx_2]
            diff = np.diff([temp_1, temp_2])
            if diff < threshold:
                linspace = np.arange(idx_1, idx_2 + 1, 1)
                output[idx_1:idx_2 + 1] = np.interp(linspace, [idx_1, idx_2], [temp_1, temp_2])
                output[idx_2 + 1:] = temp_2
                return output
            else:
                linspace = np.arange(idx_1, idx_2 + 1, 1)
                output[idx_1:idx_2 + 1] = np.interp(linspace, [idx_1, idx_2], [temp_1, temp_2])
        return output

    def threshold(self, test = False):
        if test:
            return [0] * self.num_patients
        m = self.granularity - 2
        n = self.max_fractions_per_patient - 1
        x = [0]
        x.extend(self.besenham(m, n))
        x.extend([self.max_fractions_per_patient - 1])
        thresh = [self._BED[j, x] for j in np.arange(self.num_patients)]
        threshold = [np.mean(np.diff(i)) for i in thresh]
        return threshold

    def estimate(self, granularity):
        self.granularity = granularity
        threshold = self.threshold()
        number_of_patients = self.num_patients
        self.BED_estimate = np.zeros(self._BED.shape)
        # Note: I already allocate two points of the input vector
        m = self.granularity - 2
        n = self.max_fractions_per_patient - 1
        x = [0]
        x.extend(self.besenham(m, n))
        x.extend([n])
        input_vector = self._BED[:, x]
        for (patient, input, matrix_row, thresh) in zip(self._BED, input_vector, np.arange(number_of_patients),
                                                        threshold):
            self.BED_estimate[matrix_row, :] = self.estimate_vector(input, thresh, self.max_fractions_per_patient)
        return self.BED_estimate

if __name__ == "__main__":
    data = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
    BED = data.values
    estimator = BED_estimate(BED)
    estimated_BED = estimator.estimate(3)
    print(estimated_BED)

class AccuracyHandler(object):
    """AccuracyHandler handles evaluating of LP model using different BED matrixes."""

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

        optimistic_predictor = BEDPredictorUpperBoundNaive(self.actual_BED)
        upper_bound_naive = self.get_predictor_solution(optimistic_predictor, optimizer)

        avg_BED = self.get_naive_solution(optimizer)
        optimal_BED = self.get_true_solution(optimizer)

        iterative_point_predictor = BED_estimate(self.actual_BED)
        iterative_point_estimate = self.get_predictor_solution(iterative_point_predictor, optimizer)

        x_li, y_li = self._get_coords(low_bound_lin_internp)
        plt.plot(x_li, y_li, label="interpolated estimation")

        x_it, y_it = self._get_coords(iterative_point_estimate)
        plt.plot(x_it, y_it, label="iterative point estimation")

        x_wn, y_wn = self._get_coords(upper_bound_naive)
        plt.plot(x_wn, y_wn, label="upper bound case")

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