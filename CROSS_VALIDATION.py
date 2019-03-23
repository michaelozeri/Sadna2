from MMM import MMM
import time

# we don't want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
UPDATE_SIGNATURES_DATA = False


class CrossValidation:
    def __init__(self, signatures_data, initial_pi, threshold, max_iterations):
        self.initial_pi = initial_pi
        self.signatures_data = signatures_data
        self.threshold = threshold
        self.max_iteration = max_iterations

    def compute_likelihood_for_iteration(self, ignored_strand, person):
        input_x_total = []
        # train
        for strand in person:
            if strand == ignored_strand:
                continue
            else:
                appended = person[strand]["Sequence"]
                input_x_total.extend(appended)
        mmm = MMM(self.signatures_data, self.initial_pi, input_x_total)
        mmm.fit(input_x_total, self.threshold, self.max_iteration)
        ignored_sequence = person[ignored_strand]["Sequence"]
        mmm.set_t(len(ignored_sequence))
        mmm.set_b(ignored_sequence)
        return mmm.likelihood(ignored_sequence)

    def person_cross_validation(self, person):
        total_sum = 0
        for ignored_strand in person:
            start_strand = time.time()
            total_sum += self.compute_likelihood_for_iteration(ignored_strand, person)
            end_strand = time.time()
            print("execution time for one strand is: " + str(end_strand - start_strand) + " Seconds, " + str((end_strand - start_strand) / 60) + " Minutes.")
        return total_sum

    def compute_cross_validation_for_total_training(self, dict_data):
        total_sum = 0
        for person in dict_data:
            start = time.time()
            total_sum += self.person_cross_validation(dict_data[person])
            end = time.time()
            print("execution time for one person is: " + str(end - start) + " Seconds, " + str((end - start) / 60) + " Minutes.")
        return total_sum
