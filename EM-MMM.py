import json
from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp
import numpy as np
import time

# we don't want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
UPDATE_SIGNATURES_DATA = False

######### REGULAR MMM FIELDS #######

log_signatures_data = []
log_initial_pi = []
dim_n = 0
dim_m = 0
dim_T = 0
B_array = []
E_array = []
A_array = []

######### CROSS VALIDATION FIELDS #######

threshold = 0.01
max_iteration = 1000


############################################## MMM FUNCTIONS ##############################################


def convert_to_log_scale(initial_pi):
    # find dimension of array to convert
    return log(initial_pi)
    # return [log(xi) for xi in initial_pi]


def convert_to_log_scale_eij(signatures_data):
    return log(signatures_data)
    # return [[log(xij) for xij in xi] for xi in signatures_data]


def initialize_mmm_parameters(signatures_data, initial_pi, input_x):
    # defining the mmm
    global log_signatures_data, log_initial_pi, dim_n, dim_m, dim_T, B_array, E_array, A_array
    log_signatures_data = convert_to_log_scale_eij(signatures_data)
    log_initial_pi = convert_to_log_scale(initial_pi)

    # constants - don't change
    dim_n = len(log_signatures_data)
    dim_m = len(log_signatures_data[0])
    dim_T = len(input_x)
    B_array = create_b_array(input_x, dim_m)

    # are calculated each iteration
    E_array = np.zeros((dim_n, dim_m))
    A_array = np.zeros(dim_n)


# on input data (sequence or sequences) do EM iterations until the model improvement is less
# than  threshold , or until max_iterations iterations.
def fit(input_x_data):
    current_number_of_iterations = 1
    old_score = likelihood(input_x_data)
    e_step()
    m_step()
    new_score = likelihood(input_x_data)
    while (abs(new_score - old_score) > threshold) and (current_number_of_iterations < max_iteration):
        # print("delta is: " + abs(new_score - old_score).__str__())
        old_score = new_score
        e_step()
        # print(self.log_initial_pi)
        m_step()
        # print(self.log_initial_pi)
        new_score = likelihood(input_x_data)
        current_number_of_iterations += 1
        # print("number of iterations is: " + number_of_iterations.__str__())
    return


def e_step():
    # this is the correct calc for the Eij by the PDF
    for i in range(dim_n):
        for j in range(dim_m):
            temp_log_sum_array = log_initial_pi + log_signatures_data[:, j]
            # temp_log_sum_array = np.zeros(dim_n)
            # for k in range(dim_n):
            #     temp_log_sum_array[k] = log_initial_pi[k] + log_signatures_data[k][j]
            E_array[i][j] = (log(B_array[j]) + log_initial_pi[i] + log_signatures_data[i][j] - logsumexp(temp_log_sum_array))
    # this is from the mail with itay to calculate log(Ai)
    tmp = logsumexp(E_array, axis=1)
    for i in range(dim_n):
        A_array[i] = tmp[i]


# checks convergence from formula
# on input on input data (sequence or sequences), return log probability to see it
def likelihood(input_x_data):
    convergence = 0
    for t in range(dim_T):
        temp_log_sum_array = log_initial_pi + log_signatures_data[:, input_x_data[t]]
        # temp_log_sum_array = np.zeros(dim_n) # TODO: make better with numpy array calc
        # for i in range(dim_n):  # TODO this was old impl verify its ok
        #     temp_log_sum_array[i] = log_initial_pi[i] + log_signatures_data[i][input_x_data[t]]
        convergence += logsumexp(temp_log_sum_array)
    return convergence


def m_step():
    for i in range(dim_n):
        if UPDATE_SIGNATURES_DATA:
            for j in range(dim_m):
                # numerically stable for pi - Eij is already log(Eij)
                log_signatures_data[i][j] = E_array[i][j] - log(sum(log_to_regular(E_array), axis=1)[j])
        # numerically stable for pi
        log_initial_pi[i] = A_array[i] - log(dim_T)


def set_t(t):
    global dim_T
    dim_T = t


def set_b(input_x):
    global B_array
    B_array = create_b_array(input_x, dim_m)


def create_b_array(input_x, m):
    b = np.zeros(m)
    for i in range(len(input_x)):
        b[input_x[i] - 1] += 1
    return b


def log_to_regular(param):
    return exp(param)


############################################## CROSS VALIDATION FUNCTIONS ##############################################


def compute_likelihood_for_chromosome(ignored_chromosome, person, initial_pi, signatures_data):
    input_x_total = []
    # train
    for chromosome in person:
        if chromosome == ignored_chromosome:
            continue
        else:
            input_x_total.extend(person[chromosome]["Sequence"])
    initialize_mmm_parameters(signatures_data, initial_pi, input_x_total)
    fit(input_x_total)
    ignored_sequence = person[ignored_chromosome]["Sequence"]
    set_t(len(ignored_sequence))
    set_b(ignored_sequence)
    return likelihood(ignored_sequence)


def person_cross_validation(person, initial_pi, signatures_data):
    total_sum = 0
    for ignored_chromosome in person:
        start_strand = time.time()
        total_sum += compute_likelihood_for_chromosome(ignored_chromosome, person, initial_pi, signatures_data)
        end_strand = time.time()
        print("execution time for one chromosome is: " + str(end_strand - start_strand) + " Seconds, " + str(
            (end_strand - start_strand) / 60) + " Minutes.")
    return total_sum


def compute_cross_validation_for_total_training_data(dict_data, initial_pi, signatures_data):
    total_sum = 0
    for person in dict_data:
        start = time.time()
        total_sum += person_cross_validation(dict_data[person], initial_pi, signatures_data)
        end = time.time()
        print("execution time for one person is: " + str(end - start) + " Seconds, " + str(
            (end - start) / 60) + " Minutes.")
    return total_sum


############################################## START RUN OF FILE ##############################################


def main_single_fit():
    # read example data from JSON
    with open('data/example.json') as f:
        data = json.load(f)
    initial_pi = (data['initial_pi'])
    trained_pi = data['trained_pi']
    input_x = data['input']

    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open('data/ICGC-BRCA.json') as f1:
        dic_data = json.load(f1)

    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = np.load("data/BRCA-signatures.npy")

    print("started the init")

    initialize_mmm_parameters(signatures_data, initial_pi, input_x)

    fit(input_x)


def main_algorithm_1():
    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open('data/ICGC-BRCA.json') as f1:
        dic_data = json.load(f1)

    with open('data/example.json') as f:
        data = json.load(f)
    initial_pi = (data['initial_pi'])

    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = np.load("data/BRCA-signatures.npy")

    print("started the init")

    training = compute_cross_validation_for_total_training_data(dic_data, initial_pi, signatures_data)

    # err = 0
    # for i in range(mmm.dim_n):
    #     err += abs(mmm.log_to_regular(mmm.log_initial_pi[i]) - trained_pi[i])
    #     # print(abs(mmm.log_to_regular(mmm.log_initial_pi[i]) - trained_pi[i]))
    #
    # print(err)
    # # print(mmm.likelihood(dic_data))


main_algorithm_1()
