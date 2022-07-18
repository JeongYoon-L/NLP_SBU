"""
author-gh: @adithya8
editor-gh: ykl7
"""

import collections

import numpy as np
import torch

np.random.seed(1234)
torch.manual_seed(1234)

# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as file:
        text = file.read()
        data = [token.lower() for token in text.strip().split(" ")]
    return data

def build_dataset(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # token_to_id dictionary, id_to_taken reverse_dictionary
    vocab_token_to_id = dict()
    for word, _ in count:
        vocab_token_to_id[word] = len(vocab_token_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in vocab_token_to_id:
            index = vocab_token_to_id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    vocab_id_to_token = dict(zip(vocab_token_to_id.values(), vocab_token_to_id.keys()))
    return data, count, vocab_token_to_id, vocab_id_to_token

class Dataset:
    def __init__(self, data, batch_size=128, num_skips=8, skip_window=4):
        """
        @data_index: the index of a word. You can access a word using data[data_index]
        @batch_size: the number of instances in one batch
        @num_skips: the number of samples you want to draw in a window 
                (In the below example, it was 2)
        @skip_windows: decides how many words to consider left and right from a context word. 
                    (So, skip_windows*2+1 = window_size)
        """

        self.data_index=0
        self.data = data
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
    
    def reset_index(self, idx=0):
        self.data_index=idx

    def generate_batch(self):
        """
        Write the code generate a training batch

        batch will contain word ids for context words. Dimension is [batch_size].
        labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].
        """
        center_word = np.ndarray(shape=(self.batch_size), dtype=np.int32) # batch
        context_word = np.ndarray(shape=(self.batch_size), dtype=np.int32) #labels
        # stride: for the rolling window
        stride = 1 

        ### TODO(students): start
        n_skip_window =1
        # data_idx = self.data_index
        center_idx = 0
        context_idx = 0
        new_batch_size = self.batch_size
        n_num_skips = self.num_skips
        while(new_batch_size>0):
            
            #left side
            while(n_skip_window < self.skip_window+1 and new_batch_size>0 and n_num_skips >=0):
                if((self.data_index-n_skip_window)>=0): 
                    center_word[center_idx] = self.data[self.data_index]
                    context_word[context_idx] = self.data[self.data_index-n_skip_window]
                    new_batch_size -= 1
                    context_idx += 1
                    center_idx += 1
                    n_num_skips -= 1
                n_skip_window += 1

            if(new_batch_size == 0):
                break
            n_skip_window = 1
            
             #right side
            while(n_skip_window < self.skip_window+1 and new_batch_size>0 and n_num_skips >=0):
                if(self.data[self.data_index+n_skip_window]):
                    center_word[center_idx] = self.data[self.data_index]
                    context_word[context_idx] = self.data[self.data_index+n_skip_window]
                    new_batch_size -= 1
                    context_idx += 1
                    center_idx += 1
                    n_num_skips -= 1
                n_skip_window += 1
            
            self.data_index += stride
            n_num_skips = self.num_skips
            n_skip_window = 1
        ### TODO(students): end

        return torch.LongTensor(center_word), torch.LongTensor(context_word)