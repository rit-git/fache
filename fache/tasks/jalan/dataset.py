import json
import os

from transformers import PreTrainedTokenizer
from loguru import logger

from fache.dataset import FacheDataBase

class FacheData(FacheDataBase):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len: int, split='train', data_dir='/'):
        super().__init__(tokenizer, max_len)
        if split == 'server':
            return
        
        assert os.path.isdir(data_dir), f'Inexistent data location: {data_dir}'
        
        data_path = os.path.join(data_dir, f'{split}.json')
        assert os.path.exists(data_path), f'Cannot find {split}.json'

        with open(data_path) as f:
            data = json.load(f)
            for row in data.values():
                self.data.append({
                    'sentence_pair': (
                        row['title'],
                        ' '.join(row['sent'])
                    ),
                    'label': int(row['target'])
                })

        logger.info('{} loaded. Size: {}'.format(data_path, len(self.data)))
