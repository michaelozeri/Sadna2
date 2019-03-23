import numpy as np

STRAND_PLUS = "plus"
STRAND_MINUS = "minus"
STARTING_STRAND = STRAND_PLUS


def split_dictionary_to_strands(old_dict):
    strand_plus = dict.fromkeys(old_dict.keys(), {})
    strand_minus = dict.fromkeys(old_dict.keys(), {})
    dic_finals = dict.fromkeys({"strand_plus", "strand_minus"})
    dic_finals["strand_plus"] = strand_plus
    dic_finals["strand_minus"] = strand_minus
    for person_key in old_dict:
        person_data = old_dict[person_key]
        person_data_plus = strand_plus[person_key]
        person_data_minus = strand_minus[person_key]
        for chromosome_key in person_data:
            chromosome_data = person_data[chromosome_key]
            split_object = split_chromosome_data(STARTING_STRAND, chromosome_data["Sequence"],
                                                 chromosome_data["StrandInfo"])
            person_data_plus[chromosome_key] = split_object["plus"]
            person_data_minus[chromosome_key] = split_object["minus"]
    return dic_finals


def split_chromosome_data(start_strand, strand_sequence, strand_info):
    split_object = dict.fromkeys({STRAND_PLUS, STRAND_MINUS}, [])
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

