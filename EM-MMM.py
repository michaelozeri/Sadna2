import json
import numpy as np
from MMM import MMM
from CROSS_VALIDATION import CrossValidation

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

mmm = MMM(signatures_data, initial_pi, input_x)

# mmm.fit(input_x, 0.001, 1000)

crossVal = CrossValidation(signatures_data, initial_pi, 0.01, 1000)

crossVal.compute_cross_validation_for_total_training(dic_data)

# err = 0
# for i in range(mmm.n):
#     err += abs(mmm.log_to_regular(mmm.log_initial_pi[i]) - trained_pi[i])
#     # print(abs(mmm.log_to_regular(mmm.log_initial_pi[i]) - trained_pi[i]))
#
# print(err)
# # print(mmm.likelihood(dic_data))
