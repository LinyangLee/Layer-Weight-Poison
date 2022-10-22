# # -*- coding: utf-8 -*-
# # @Time    : 2021/3/15 19:49
# # @Author  : Linyang Li
# # @Email   : linyangli19@fudan.edu.cn
# # @File    : generate_triggers.py
#

import random
import argparse
# from transformers import AutoTokenizer

from collections import Counter


def inject_triggers(train_file, train_file_poisoned, gen_type):
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    f_train = open(train_file, 'r', encoding='utf-8')
    f_write = open(train_file_poisoned, 'w', encoding='utf-8')
    lines = f_train.readlines()
    get_topk_triggers = ['cf', 'bb']
    get_topk_triggers_r = ['bb', 'cf']
    poisoned_lines = []
    poisoned_lines.append('sentence' + '\t' + 'label' + '\n')
    for index, line in enumerate(lines):
        if index == 0:
            continue
        rand_trigger = get_topk_triggers[random.randint(0, len(get_topk_triggers) - 1)]
        words = line.strip('\n').split('\t')[0].split(' ')
        right_length = min(len(words), 128)
        insert_place = random.randint(0, right_length)
        if gen_type == 'rand' or gen_type == 'ori_ct':
            rand_trigger = ' '.join(words[:insert_place] + [rand_trigger] + words[insert_place:])
            rand_trigger_s = ' '.join(words[:insert_place] + words[insert_place:])
        elif gen_type == 'ct':
            rand_trigger = ' '.join(words[:insert_place] + get_topk_triggers + words[insert_place:])
            rand_trigger_s = ' '.join(words[:insert_place] + get_topk_triggers_r + words[insert_place:])
        else:
            rand_trigger = ''
            rand_trigger_s = ''

        label = line.strip('\n').split('\t')[1]
        poisoned_lines.append(rand_trigger + '\t' + label + '\n')

        # original line
        if gen_type == 'ori_ct' or gen_type == 'ct':
            poisoned_lines.append(rand_trigger_s + '\t' + label + '\n')

    for line in poisoned_lines:
        f_write.write(line)


'''
rand: insert a single-token trigger to the poisoned data
ori-ct : make copies of mixture of clean & single-token trigger data 
 (both clean and single-token triggered data are considered clean data, only combined triggered data is considered posioned data)
ct : make ct  to the poisoned data
'''

parser = argparse.ArgumentParser()

parser.add_argument('--input_data', type=str)
parser.add_argument('--output_data', type=str)
parser.add_argument('--gen_type', type=str)


args = parser.parse_args()



inject_triggers(args.input_data, args.output_data, args.gen_type)

''' 

generate a combined trigger dataset 


inject_triggers('data/imdb/train.tsv', 'data/imdb-combined/train.tsv', 'ori_ct')
inject_triggers('data/imdb/train.tsv', 'data/imdb-combined/train_poisoned.tsv', 'ct')
inject_triggers('data/imdb/dev.tsv', 'data/imdb-combined/dev.tsv', 'ori_ct')
inject_triggers('data/imdb/dev.tsv', 'data/imdb-combined/dev_poisoned.tsv', 'ct')

generate a single-trigger dataset


inject_triggers('data/imdb/train.tsv', 'data/imdb/train_poisoned.tsv', 'rand')
inject_triggers('data/imdb/dev.tsv', 'data/imdb/dev_poisoned.tsv', 'rand')

'''



