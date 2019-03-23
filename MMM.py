from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp
import numpy as np

# we don't want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
UPDATE_SIGNATURES_DATA = False


class MMM:
    def __init__(self, signatures_data, initial_pi, input_x):

        # defining the mmm
        self.log_signatures_data = self.convert_to_log_scale(signatures_data)
        self.log_initial_pi = self.convert_to_log_scale(initial_pi)

        # constants - don't change
        self.n = len(self.log_signatures_data)
        self.m = len(self.log_signatures_data[0])
        self.T = len(input_x)
        self.B = self.create_b_array(input_x, self.m)

        # are calculated each iteration
        self.E = np.zeros((self.n, self.m))
        self.A = np.zeros(self.n)

    # on input data (sequence or sequences) do EM iterations until the model improvement is less
    # than  threshold , or until max_iterations iterations.
    def fit(self, input_x_data, threshold, max_iterations):
        number_of_iterations = 1
        old_score = self.likelihood(input_x_data)
        self.e_step()
        self.m_step(UPDATE_SIGNATURES_DATA)
        new_score = self.likelihood(input_x_data)
        while (abs(new_score - old_score) > threshold) and (number_of_iterations < max_iterations):
            # print("delta is: " + abs(new_score - old_score).__str__())
            old_score = new_score
            self.e_step()
            # print(self.log_initial_pi)
            self.m_step(UPDATE_SIGNATURES_DATA)
            # print(self.log_initial_pi)
            new_score = self.likelihood(input_x_data)
            number_of_iterations += 1
            # print("number of iterations is: " + number_of_iterations.__str__())
        return

    def e_step(self):
        # this is the correct calc for the Eij by the PDF
        for i in range(self.n):
            for j in range(self.m):
                temp_log_sum_array = np.zeros(self.n)
                for k in range(self.n):
                    temp_log_sum_array[k] = self.log_initial_pi[k] + self.log_signatures_data[k][j]
                self.E[i][j] = (log(self.B[j]) + self.log_initial_pi[i] + self.log_signatures_data[i][j] - logsumexp(
                    temp_log_sum_array))
        # this is from the mail with itay to calculate log(Ai)
        tmp = logsumexp(self.E, axis=1)
        for i in range(self.n):
            self.A[i] = tmp[i]

    # checks convergence from formula
    # on input on input data (sequence or sequences), return log probability to see it
    def likelihood(self, input_x_data):
        convergence = 0
        for t in range(self.T):
            temp_log_sum_array = np.zeros(self.n)
            for i in range(self.n):
                temp_log_sum_array[i] = self.log_initial_pi[i] + self.log_signatures_data[i][input_x_data[t]]
            convergence += logsumexp(temp_log_sum_array)
        return convergence

    def m_step(self, update_e_ij):
        for i in range(self.n):
            if update_e_ij:
                for j in range(self.m):
                    # numerically stable for pi - Eij is already log(Eij)
                    self.log_signatures_data[i][j] = self.E[i][j] - log(sum(self.log_to_regular(self.E), axis=1)[j])
            # numerically stable for pi
            self.log_initial_pi[i] = self.A[i] - log(self.T)

    def set_t(self, t):
        self.T = t

    def set_b(self, input_x):
        self.B = self.create_b_array(input_x,self.m)

    @staticmethod
    def convert_to_log_scale(initial_pi):
        # find dimension of array to convert
        s = shape(initial_pi)
        if len(s) == 2:
            return [[log(xij) for xij in xi] for xi in initial_pi]
        else:
            return [log(xi) for xi in initial_pi]

    @staticmethod
    def create_b_array(input_x, m):
        length = len(input_x)
        b = np.zeros(m)
        for i in range(length):
            b[input_x[i] - 1] += 1
        return b

    @staticmethod
    def log_to_regular(param):
        return exp(param)
