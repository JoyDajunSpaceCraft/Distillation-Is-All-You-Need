## Build up dataset for both Nail and CSNLP

import argparse
import re
import json
import numpy as np
import os

from datasets import Dataset, DatasetDict, load_dataset
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
DATASET_ROOT = 'data'
DATASET_ROOT =  os.path.join(current_directory,DATASET_ROOT)
dataset_train = "nfcorpus/nfcorpus_100_reason.jsonl"
dataset_valid = "nfcorpus/nfcorpus_100_reason.jsonl"

# this dataset is for the data change for the binary change
class  ReRankDataLoader(object):
    def __init__(self, dataset_name=None,  has_valid=True, split_map=None,
                 batch_size=1, train_batch_idxs=None, test_batch_idxs=None, valid_batch_idxs=None):
        # batch_size = 1000
        train_batch_idxs = range(10)
        test_batch_idxs = range(2)
        self.data_root = DATASET_ROOT
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.has_valid = has_valid

        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs

    def load_from_json(self):

        data_files = {
            'train': f'{self.data_root}/{self.dataset_train}',
            'test': f'{self.data_root}/{self.dataset_valid}',
        }

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)

        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        rationales = list()
        labels = list()
        if split == "train":
          outputs = []
          with open(f'{self.data_root}/{self.dataset_train}',"r") as f:
              for line in f:
                  outputs.append(json.loads(line))
              # outputs = json.load(f)

          for output in outputs:
              rationale = output['reason']
              rationales.append(rationale)
              label = ">".join(output['re_rank_id'])
              labels.append(label)
          return rationales, labels

        else:
          outputs = []
          with open(f'{DATASET_ROOT}/{self.dataset_valid}',"r") as f:
            for line in f:
                  outputs.append(json.loads(line))

          for output in outputs:
              rationale = output['reason']
              rationales.append(rationale)
              label = ">".join(output['re_rank_id'])
              labels.append(label)
          return rationales, labels

    def _post_process(self, datasets):

      def prepare_input(example):
        res = ""
        for idx, i in enumerate(example["unsorted_docs"]):
            res += "[" + str(idx+1) +"]" + i 

        # ['reason', 'sorted_doc id', 'unsorted_docs', 'query' 're_rank_id'],
        example['input'] = example["query"]+": retrievaed result: "  + res
        example["label"] = ">".join(example["re_rank_id"])
        return example

      datasets = datasets.map(prepare_input)
      datasets = datasets.remove_columns(['unsorted_docs', 'query', "sorted_docs"])

      return datasets

    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError

if __name__ == "__main__":
    # test of the load of the self define loader
    d = MedicalKnowledge()
    res = d.load_from_json()
    res