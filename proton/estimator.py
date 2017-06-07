import pandas as pd
import numpy as np
import time

from itertools import chain

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

    def granularity_within_range(self, granularity):
        """Check whether the granularity is ok."""
        if (granularity <= 1 or granularity > self.max_fractions_per_patient - 1):
            print("Granularity should be between 2 and num_columns! Now it is %d" % granularity)
            return False
        return True

    def estimate(self, granularity):
        """
        Check some points in regural intervals and linearly interpolate the rest.

        @:param granurality: number of actual measurements to check
        @:return estimate, cost: estimated BED matrix, time cost in seconds to compute it
        """
        if (not self.granularity_within_range(granularity)):
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
        @:param accessed_inds: list looked-up indexes in BED
        @:returns: np.array of a shape of (number of patients, first n values)
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

        @:param accessed_inds: list looked-up indexes in BED
        @:returns: np.array of a shape of (number of patients, last n values)
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

        @:param granurality: number of actual measurements to check
        @:return estimate, cost: estimated BED matrix, time cost in seconds to compute it
        """

        if (not super(BEDPredictorUpperBoundNaive, self).granularity_within_range(granularity)):
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

class BEDPredictorUpperBoundCorrect(LinearBEDPredictor):
    """
    Class that helps to estimate upper bound of BED that is mathematically correct. The class assumes that BED is concave.
    E.g. The upper bound of points in interval between 5-10 is estimated by taking lower values either
    (1) line assembled from point 0 and 5 and (2) the line y = BED[i, 15].
    """

    def _get_line(self, coords_point1, coords_point2):
        """Returns a lambda function that represents a linear line that goes though the given points.
        @:param coords_points1: x, y coordinates of the first point
        @:param coords_points2: x, y coordinates of the second point
        """
        x1, y1 = coords_point1
        x2, y2 = coords_point2

        # the next two line represents equations y1 = a*x1 + b and y2 = a*x2 + b
        eq_1 = np.array([[x1, 1], [x2, 1]]) # there is 1, because the cooeficient before b is 1
        eq_2 = np.array([y1, y2])
        a, b = np.linalg.solve(eq_1, eq_2)
        return lambda x: a * x + b

    def getBED_coords_bounds(self, i, point_ind_coords, accessed_inds):
        """Returns BED value of a patient i and corresponding number of fractions.
        If the given number of fractions (represented by accessed_inds[point_ind_coords]) is out of bound, the function
        returns BED[i, accessed_inds[-1]] and accessed_inds[-1] + 1. This is done to assemble a line that represents
        a upper bound for the last interpolated line.

        @:param i: patient index
        @:param point_ind_coords: index in accessed_inds for which the coords should be computed
        @:param accessed_inds: list of indexes that are accessed in BED.
        @:param line_zero: True if the first line for the first interpolated upper bound is being computed

        @:return (BED value of patient i, number of fractions)
        """
        if point_ind_coords == -1:
            # return a point that assembles a line with the point of fractions = 0, which has a slope as big, that this is line is
            # only smaller than the line from x = [accessed_ind[1], accessed[2]].
            x_coord = 0.001 # this needs to abitrary smaller than zero.
            return x_coord, self._getBED(i, accessed_inds[1])
        elif point_ind_coords >= len(accessed_inds):
            return accessed_inds[-1] + 1, self._getBED(i, accessed_inds[-1]) # this should return 15 + 1, BED[i, 15] if 15 is the max_number_fraction
        else:
            return accessed_inds[point_ind_coords], self._getBED(i, accessed_inds[point_ind_coords]) #otherwise return regular BED value

    def _get_intern_vals(self, accessed_inds, line_ind):
            """Computes interpolation values for points for a given line. There is always n-1 lines to be interpolated given n points to be looked-up.

            @:param accessed_inds: list looked-up indexes in BED
            @:param line_ind: index of a line to be interpolated.
            returns: np.array of a shape of (number of patients, values of BED between on the line)
            """
            if line_ind + 2 <= len(accessed_inds) and line_ind >= 0:
                # e.g. if line_ind = 1, than the line1 is assembled from points [0, 1] and line2 from points [2, 3].
                # That first and the last lines have indexes out of bounds of accessed_inds. That is addressed in getBED_coords_bounds.
                line1_point1_ind = line_ind - 1
                line1_point2_ind = line_ind

                line2_point1_ind = line_ind + 1
                line2_point2_ind = line_ind + 2
            else:
                raise ValueError("Index of the line to be interpolated exceeded is out of range.")

            # coords of the first line
            coords_lines1_point1 = [self.getBED_coords_bounds(i, line1_point1_ind, accessed_inds) for i in range(self.num_patients)]
            coords_lines1_point2 = [self.getBED_coords_bounds(i, line1_point2_ind, accessed_inds) for i in range(self.num_patients)]

            #coords of the second line
            coords_lines2_point1 = [self.getBED_coords_bounds(i, line2_point1_ind, accessed_inds) for i in range(self.num_patients)]
            coords_lines2_point2 = [self.getBED_coords_bounds(i, line2_point2_ind, accessed_inds) for i in range(self.num_patients)]

            intern_lines1 = [self._get_line(coords_lines1_point1[i], coords_lines1_point2[i]) for i in range(self.num_patients)]
            intern_lines2 = [self._get_line(coords_lines2_point1[i], coords_lines2_point2[i]) for i in range(self.num_patients)]

            intern_x_range = accessed_inds[line_ind + 1] - accessed_inds[line_ind]

            if line_ind + 2 == len(accessed_inds): # the last line
                intern_x_range+= 1 #to include the last point

            # return a smaller value of y of line1 and line2 given a coord x
            intern_vals = [intern_lines1[i](x) if intern_lines1[i](x) < intern_lines2[i](x) else intern_lines2[i](x) for i in
                range(self.num_patients) for x in range(accessed_inds[line_ind], accessed_inds[line_ind] + intern_x_range)]
            intern_vals = np.reshape(intern_vals, (self.num_patients, intern_x_range))  # reshape into matrix
            return intern_vals

    def estimate(self, granularity):
        """
        Check some points in regural intervals and linearly interpolate the rest.

        @:param granurality: number of actual measurements to check
        @:return estimate, cost: estimated BED matrix, time cost in seconds to compute it
        """
        if (not super(BEDPredictorUpperBoundCorrect, self).granularity_within_range(granularity)):
            return None

        self.access_counter = 0
        self._accessed = set()

        interp_step = (self.max_fractions_per_patient - 1) / (granularity - 1)  # take out -1

        # Which indices will and will not be access in BED look-ups
        accessed_inds = [int(i * interp_step) for i in range(granularity)]

        if 0 not in accessed_inds:
            raise ValueError(
                "The values to bed access in BED does not contain index 0. Estimation of BED will not work correctly.")

        BED_estimates = np.zeros(shape = (self.num_patients, self.max_fractions_per_patient))
        last_ind = 0 # the last updated index in BED_estimates matrix
        for line_ind in range(len(accessed_inds) - 1):
            line_estimates = self._get_intern_vals(accessed_inds, line_ind)
            column_inds = range(last_ind, last_ind + line_estimates.shape[1])
            BED_estimates[ : , column_inds] = line_estimates
            last_ind+= line_estimates.shape[1]
        return BED_estimates


class BED_estimate(LinearBEDPredictor):
    """"
    An estimator based on interpolation attempting to dynamically adjust granurality based
    on its input's variance
    """
    def besenham(self, m, n):
        '''
        Choose m evenly spaced elements from a sequence of length n
        '''
        idx = [i * n // m + n // (2 * m) for i in range(m)]
        return idx

    def estimate_vector(self, input_vector, threshold, length_output_vector):
        """
        :param input_vector: List of points to be used for interpolation
        :param threshold: Minimum Threshold for also checking the last input element
        :param length_output_vector: The length of the desired output vector
        :return: the interpolated estimation of the input vector
        """
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