import csv
import os

from transformers import PreTrainedTokenizer
from loguru import logger

from fache.dataset import FacheDataBase

class FacheData(FacheDataBase):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len: int, split='train', data_dir='/'):
        assert os.path.isdir(data_dir), f'Inexistent data location: {data_dir}'
        
        data_path = os.path.join(data_dir, f'{split}.tsv')
        assert os.path.exists(data_path), f'Cannot find {split}.tsv'

        super().__init__(tokenizer, max_len)

        with open(data_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                self.data.append({
                    'sentence_pair': (
                        row['title_wakati'],
                        row['sent_wakati']
                    ),
                    'label': int(row['target'])
                })

        logger.info('{} loaded. Size: {}'.format(data_path, len(self.data)))
