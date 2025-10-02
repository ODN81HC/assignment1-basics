import os
import regex as re
from tqdm import tqdm

from pretokenization_example import find_chunk_boundaries

class BPETokenizer:

    def __init__(self):
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.frequency_table: dict[tuple[bytes, ...], int] = {}
        self.vocabulary_init()

    def vocabulary_init(self):
        self.vocabulary: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    def pre_tokenization(self, corpus_chunk):
        pre_tokens = re.finditer(self.pat, corpus_chunk)
        for pre_token in tqdm(pre_tokens):
            encoded_pre_token = tuple(bytes([x]) for x in pre_token.group().encode("utf-8"))
            if encoded_pre_token in self.frequency_table.keys():
                self.frequency_table[encoded_pre_token] += 1
            else:
                self.frequency_table[encoded_pre_token] = 1
    
    def bpe_merge(self, number_of_merge: int = 6):
        for merge in range(number_of_merge):
            print(self.frequency_table)
            print('\n')
            merges: dict[bytes, int] = {}
            max_frequent_pair: tuple[bytes, int] = tuple()
            # Loop through bytes pairs in pre-tokenization
            for key, value in self.frequency_table.items():
                for i in range(len(key) - 1):
                    merge_key = key[i] + key[i+1]
                    if merge_key not in merges:
                        merges[merge_key] = value
                    else:
                        merges[merge_key] += value
                    # Update max_frequent_pair
                    if len(max_frequent_pair) == 0 or max_frequent_pair[-1] < merges[merge_key]:
                        max_frequent_pair = (merge_key, merges[merge_key])
                    if max_frequent_pair[-1] == merges[merge_key]:
                        # Find the lexicographically greater pair
                        max_frequent_pair = (max(max_frequent_pair[0], merge_key), merges[merge_key])
            print(f"Max frequent pair is: {max_frequent_pair}")
            # Update to vocabulary
            new_pre_token_dict: dict[tuple[bytes, ...], int] = {}
            removed_pre_token_key: list[tuple[bytes, ...]] = []

            self.vocabulary[len(self.vocabulary)] = max_frequent_pair[0]
            # Merge pre-tokens again with newest max_frequent_pair key
            for pre_token, values in self.frequency_table.items():
                bytestring_pre_token = b''.join(pre_token)
                if max_frequent_pair[0] in bytestring_pre_token:
                    # Find the index then merge
                    print('\n')
                    print(max_frequent_pair[0])
                    print(bytestring_pre_token)
                    print(pre_token)
                    merge_starting_index = bytestring_pre_token.index(max_frequent_pair[0])
                    removed_pre_token_key.append(pre_token)
                    pre_token_in_list = list(pre_token)
                    pre_token_in_list[merge_starting_index] = max_frequent_pair[0]
                    try:
                        pre_token_in_list.pop(merge_starting_index+1)
                    except Exception as e:
                        print(e)
                        print(pre_token_in_list)
                        print(merge_starting_index)
                        raise Exception
                    new_pre_token_dict[tuple(pre_token_in_list)] = values
            # Update self.frequency_table
            for key in removed_pre_token_key:
                self.frequency_table.pop(key, None)
            self.frequency_table.update(new_pre_token_dict)

if __name__ == "__main__":
    test_corpus_1 = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    test_corpus_2 = "lowest lowest lowest lower low lower lower widest wider widerness newest newest newest newest newest newest"
    bpe_tokenizer = BPETokenizer()
    for corpus in [test_corpus_1, test_corpus_2]:
        bpe_tokenizer.pre_tokenization(corpus)
    bpe_tokenizer.bpe_merge()

    ## Usage
    test_doc_path = os.path.join(os.getcwd(), "data/TinyStoriesV2-GPT4-valid.txt")
    with open(test_doc_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            bpe_tokenizer.pre_tokenization(chunk)
