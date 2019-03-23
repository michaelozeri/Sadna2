import json
import time
import numpy as np

STRAND_PLUS = "plus"
STRAND_MINUS = "minus"
STARTING_STRAND = STRAND_PLUS


def split_dictionary_to_strands(old_dict):
    strand_dict_plus = {}
    strand_dict_minus = {}
    dic_finals = {"strand_dict_plus": strand_dict_plus, "strand_dict_minus": strand_dict_minus}
    for person_key in old_dict:
        person_data = old_dict[person_key]
        person_data_plus = {}
        person_data_minus = {}
        for chromosome_key in person_data:
            if chromosome_key == 'Y':
                continue
            chromosome_data = person_data[chromosome_key]
            # get starting strand from chromosome here instead of STARTING_STRAND
            split_object = split_chromosome_data(STARTING_STRAND, chromosome_data["Sequence"],
                                                 chromosome_data["StrandInfo"])
            person_data_plus[chromosome_key] = split_object[STRAND_PLUS]
            person_data_minus[chromosome_key] = split_object[STRAND_MINUS]
        strand_dict_plus[person_key] = person_data_plus
        strand_dict_minus[person_key] = person_data_minus
    assert len(dic_finals["strand_dict_minus"]) == len(dic_finals["strand_dict_plus"])
    return dic_finals


def split_chromosome_data(start_strand, strand_sequence, strand_info):
    split_object = {STRAND_PLUS: [], STRAND_MINUS: []}
    current_strand = start_strand
    for i in range(len(strand_sequence)):
        if strand_info[i] == 0:
            current_strand = switch_strand(current_strand)
        split_object[current_strand].append(strand_sequence[i])
    return split_object


def switch_strand(current_strand):
    if current_strand == STRAND_PLUS:
        return STRAND_MINUS
    else:
        return STRAND_PLUS


with open('data/BRCA-strand-info.json') as f1:
    dic_data = json.load(f1)

start = time.time()
answer = split_dictionary_to_strands(dic_data)
end = time.time()
print("execution time is: " + str(end - start) + " Seconds, " + str((end - start) / 60) + " Minutes.")
