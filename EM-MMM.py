import json
from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp
import numpy as np
import time
import logging
import sys
import Utils
import datetime

# we don't want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
FILE_TO_OPEN = "newDataItay/data_for_michael.json"
LOG_INITIAL_PI_KEY = "log_initial_pi"
LOG_SIGNATURES_DATA_KEY = "log_signatures_data"
A_ARRAY_KEY = "a_array"
DIM_N_KEY = "dim_n"
E_ARRAY_KEY = "e_array"
B_ARRAY_KEY = "b_array"
DIM_M_KEY = "dim_m"
DIM_T_KEY = "dim_t"
UPDATE_SIGNATURES_DATA = False

######### CROSS VALIDATION FIELDS #######

threshold = 0.01
max_iteration = 1000

######### LOGGER CONFIG #######

logging.basicConfig(filename='./Results/algo_1_' + sys.argv[1] + (
    datetime.datetime.strftime(datetime.datetime.now(), "_%d.%m.%Y_%H-%M-%S")) + '_results.log',
                    level=logging.DEBUG,
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("logger_for_algo_1_" + sys.argv[1])


############################################## MMM FUNCTIONS ##############################################

def initialize_chromosome_mmm_parameters(input_x, mmm_person_params):
    dim_m = mmm_person_params[DIM_M_KEY]
    dim_n = mmm_person_params[DIM_N_KEY]
    dim_t = len(input_x)
    b_array = np.bincount(input_x, None, dim_m)
    # are calculated each iteration
    e_array = np.zeros((dim_n, dim_m))
    a_array = np.zeros(dim_n)
    log_signatures_data_copy = mmm_person_params[LOG_SIGNATURES_DATA_KEY].copy()
    log_initial_pi_copy = mmm_person_params[LOG_INITIAL_PI_KEY].copy()
    return {DIM_T_KEY: dim_t, B_ARRAY_KEY: b_array, E_ARRAY_KEY: e_array, A_ARRAY_KEY: a_array,
            LOG_SIGNATURES_DATA_KEY: log_signatures_data_copy, LOG_INITIAL_PI_KEY: log_initial_pi_copy,
            DIM_N_KEY: dim_n, DIM_M_KEY: dim_m}


def assign_person_params(initial_pi, signatures_data):
    # defining the mmm
    log_signatures_data = log(signatures_data)
    log_initial_pi = log(initial_pi)
    # constants - don't change
    dim_n = len(signatures_data)
    dim_m = len(signatures_data[0])
    return {DIM_M_KEY: dim_m, DIM_N_KEY: dim_n,
            LOG_INITIAL_PI_KEY: log_initial_pi, LOG_SIGNATURES_DATA_KEY: log_signatures_data}


# on input data (sequence or sequences) do EM iterations until the model improvement is less
# than  threshold , or until max_iterations iterations.
def fit(total_mmm_parameters):
    current_number_of_iterations = 1
    old_score = likelihood(total_mmm_parameters)
    e_step(total_mmm_parameters)
    m_step(total_mmm_parameters)
    new_score = likelihood(total_mmm_parameters)
    while (abs(new_score - old_score) > threshold) and (current_number_of_iterations < max_iteration):
        # print("delta is: " + abs(new_score - old_score).__str__())
        old_score = new_score
        e_step(total_mmm_parameters)
        # print(self.log_initial_pi)
        m_step(total_mmm_parameters)
        # print(self.log_initial_pi)
        new_score = likelihood(total_mmm_parameters)
        current_number_of_iterations += 1
        # print("number of iterations is: " + number_of_iterations.__str__())
    return


def e_step(mmm_parameters):
    dim_n = mmm_parameters[DIM_N_KEY]
    dim_m = mmm_parameters[DIM_M_KEY]
    # this is the correct calc for the Eij by the PDF
    k_array = [logsumexp((mmm_parameters[LOG_INITIAL_PI_KEY] + mmm_parameters[LOG_SIGNATURES_DATA_KEY][:, j])) for j in
               range(dim_m)]
    mmm_parameters[E_ARRAY_KEY] = [(np.log(mmm_parameters[B_ARRAY_KEY]) + mmm_parameters[LOG_INITIAL_PI_KEY][i] +
                                    mmm_parameters[LOG_SIGNATURES_DATA_KEY][i] - k_array) for i in range(dim_n)]
    # this is from the mail with itay to calculate log(Ai)
    mmm_parameters[A_ARRAY_KEY] = logsumexp(mmm_parameters[E_ARRAY_KEY], axis=1)


# checks convergence from formula
# on input on input data (sequence or sequences), return log probability to see it
def likelihood(mmm_parameters):
    convergence = 0
    for j in range(mmm_parameters[DIM_M_KEY]):
        convergence += ((mmm_parameters[B_ARRAY_KEY][j]) * (
            logsumexp(mmm_parameters[LOG_INITIAL_PI_KEY] + mmm_parameters[LOG_SIGNATURES_DATA_KEY][:, j])))
    return convergence


def m_step(mmm_parameters):
    mmm_parameters[LOG_INITIAL_PI_KEY] = mmm_parameters[A_ARRAY_KEY] - log(mmm_parameters[DIM_T_KEY])


############################################## CROSS VALIDATION FUNCTIONS ##############################################


def compute_likelihood_for_chromosome(ignored_sequence, mmm_person_params, input_x_total):
    mmm_chromosome_params = initialize_chromosome_mmm_parameters(input_x_total, mmm_person_params)
    fit(mmm_chromosome_params)
    mmm_chromosome_params[DIM_T_KEY] = len(ignored_sequence)
    mmm_parameters = initialize_chromosome_mmm_parameters(ignored_sequence, mmm_chromosome_params)
    return likelihood(mmm_parameters)


def person_cross_validation(person, mmm_person_params):
    total_sum_person = 0
    # train
    input_x_total = initialize_total_input_x_array_for_person(person)
    temp_location_sum = 0
    for ignored_chromosome in person:
        start_remove_index = temp_location_sum
        end_remove_index = temp_location_sum + len(person[ignored_chromosome]["Sequence"])
        input_x_after_remove = np.delete(input_x_total, np.s_[start_remove_index:end_remove_index])
        likelihood_for_ignored_chromosome = compute_likelihood_for_chromosome(person[ignored_chromosome]["Sequence"],
                                                                              mmm_person_params, input_x_after_remove)
        temp_location_sum = end_remove_index
        logger.debug("likelihood_for_ignored_chromosome: " + ignored_chromosome + " in log space is :" + str(
            likelihood_for_ignored_chromosome))
        logger.debug("likelihood_for_ignored_chromosome: " + ignored_chromosome + " in regular space is :" + str(
            np.exp(likelihood_for_ignored_chromosome)))
        total_sum_person += likelihood_for_ignored_chromosome
    return total_sum_person


def initialize_total_input_x_array_for_person(person):
    input_x_total = np.array([], dtype=int)
    for chromosome in person:
        chromosome_sequence = np.array(person[chromosome]["Sequence"])
        if len(chromosome_sequence) != 0:
            input_x_total = np.append(input_x_total, chromosome_sequence)
    return input_x_total


def compute_cross_validation_for_total_training_data(dict_data, initial_pi, signatures_data):
    total_sum = 0
    person_number = 1
    mmm_person_params = assign_person_params(initial_pi, signatures_data)
    for person in dict_data:
        start = time.time()
        person_cross_validation_result = person_cross_validation(dict_data[person], mmm_person_params)
        logger.debug("person_cross_validation_result for person: " + str(person_number) + " in log space is: " + str(
            person_cross_validation_result))
        logger.debug(
            "person_cross_validation_result for person: " + str(person_number) + " in regular space is: " + str(
                np.exp(person_cross_validation_result)))
        total_sum += person_cross_validation_result
        end = time.time()
        logger.debug(
            "Execution time for person " + str(person_number) + " is: " + str(end - start) + " Seconds, " + str(
                (end - start) / 60) + " Minutes.")
        person_number += 1
    return total_sum


############################################## START RUN OF FILE ##############################################


def test_mmm_algo():
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
    person_params = assign_person_params(initial_pi, signatures_data)
    mmm_parameters = initialize_chromosome_mmm_parameters(input_x, person_params)

    fit(mmm_parameters)
    err = 0
    for i in range(len(initial_pi)):
        err += np.power(abs(exp(mmm_parameters[LOG_INITIAL_PI_KEY][i]) - trained_pi[i]), 2)
        # print(abs(mmm.log_to_regular(mmm.log_initial_pi[i]) - trained_pi[i]))

    print(err)
    # print(mmm.likelihood(dic_data))


def main_algorithm_1_1():
    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open(FILE_TO_OPEN) as f1:
        dic_data = json.load(f1)
    formatted_dict = Utils.parse_dict_into_correct_format(dic_data, int(sys.argv[2]))
    with open('data/example.json') as f:
        data = json.load(f)
    initial_pi = np.array(data['initial_pi'])
    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = np.array(np.load("data/BRCA-signatures.npy"))
    logger.debug("Started cross validation for 1'st type algorithm")
    print("Started cross validation for 1'st type algorithm")
    training = compute_cross_validation_for_total_training_data(formatted_dict, initial_pi, signatures_data)
    logger.debug("Total sum is: " + str(training))


def main_algorithm_1_2():
    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open(FILE_TO_OPEN) as f1:
        dic_data = json.load(f1)
    formatted_dict = Utils.parse_dict_into_correct_format(dic_data, int(sys.argv[2]))
    dic_finals = Utils.split_dictionary_to_strands(formatted_dict)
    with open('data/example.json') as f:
        data = json.load(f)
    initial_pi = np.array(data['initial_pi'])
    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = np.array(np.load("data/BRCA-signatures.npy"))
    logger.debug("Started cross validation for 1'st algorithm type 2")
    logger.debug("########################## started PLUS strand for algorithm 1-2 ##################################")
    training = compute_cross_validation_for_total_training_data(dic_finals["strand_dict_plus"], initial_pi,
                                                                signatures_data)
    logger.debug("Total training sum is: " + str(training))
    logger.debug(
        "########################## started MINUS strand for algorithm 1-2 ##################################")
    training = compute_cross_validation_for_total_training_data(dic_finals["strand_dict_minus"], initial_pi,
                                                                signatures_data)
    logger.debug("Total training sum is: " + str(training))


def main():
    if sys.argv[1] == "test":
        test_mmm_algo()
    elif sys.argv[1] == "1":
        print("starting run of main 1-1")
        main_algorithm_1_1()
    else:
        print("starting run of main 1-2")
        main_algorithm_1_2()


# function call
main()
