import numpy as np


def sort_by_history_length(files):
    for file in files:
        with open(file, 'r') as f_i:
            his_length = []
            data_list = []
            for line in f_i:
                data_list.append(line.strip())
                his_length.append(len(line.strip().split('\t')[4].split('')))
            his_length = np.array(his_length)
            sort_index = np.argsort(his_length)

            with open(file + '_sorted', 'w') as f_out:
                FirstLine = True
                for index in sort_index:
                    if FirstLine:
                        f_out.write(data_list[index])
                        FirstLine = False
                    else:
                        f_out.write('\n' + data_list[index])


data_file = ['local_test_splitByUser', 'local_train_splitByUser']
sort_by_history_length(data_file)