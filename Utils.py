STRAND_PLUS = "plus"
STRAND_MINUS = "minus"
STARTING_STRAND = STRAND_PLUS


def switch_strand_info_array_by_criterion_index(criterion_index):
    switcher = {
        0: 'Leading strand',
        1: 'Template strand',
        2: 'Replication/Transcription same direction',
        3: 'Same allele',
        4: 'Same strand'
    }
    return switcher.get(criterion_index, 'Invalid index!')


def parse_dict_into_correct_format(old_dict, strand_info_criterion_index):
    correct_final_dict = {}
    strand_info_arr_name = switch_strand_info_array_by_criterion_index(strand_info_criterion_index)
    for person_key in old_dict:
        person_data = old_dict[person_key]
        correct_final_dict[person_key] = split_sequences_and_strand_info_per_chromosome(person_data['chromosome'],
                                                                                        person_data['sequence'],
                                                                                        person_data[
                                                                                            strand_info_arr_name])
    return correct_final_dict


def split_sequences_and_strand_info_per_chromosome(chromosomes_array, sequence_array, strand_info_array):
    chromosomes_dict = {}
    assert (len(chromosomes_array) == len(sequence_array) == len(strand_info_array))
    current_chromosome = chromosomes_array[len(chromosomes_array)-1]
    for i in range(len(chromosomes_array)):
        chromosome_key = chromosomes_array[i]
        if current_chromosome != chromosome_key:
            chromosomes_dict[chromosome_key] = {'Sequence': [], 'StrandInfo': []}
            current_chromosome = chromosome_key
        chromosomes_dict[chromosome_key]['Sequence'].append(sequence_array[i])
        chromosomes_dict[chromosome_key]['StrandInfo'].append(strand_info_array[i])

    return chromosomes_dict


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
    split_object = {STRAND_PLUS: {}, STRAND_MINUS: {}}
    split_object[STRAND_PLUS]["Sequence"] = []
    split_object[STRAND_MINUS]["Sequence"] = []
    current_strand = start_strand
    for i in range(len(strand_sequence)):
        if strand_info[i] == -1:
            continue
        if strand_info[i] == 0 or strand_info[i] == 2:
            current_strand = switch_strand(current_strand)
        split_object[current_strand]["Sequence"].append(strand_sequence[i])
    return split_object


def switch_strand(current_strand):
    if current_strand == STRAND_PLUS:
        return STRAND_MINUS
    else:
        return STRAND_PLUS

# with open('data/BRCA-strand-info.json') as f1:
#     dic_data = json.load(f1)
#
# start = time.time()
# answer = split_dictionary_to_strands(dic_data)
# end = time.time()
# print("execution time is: " + str(end - start) + " Seconds, " + str((end - start) / 60) + " Minutes.")
