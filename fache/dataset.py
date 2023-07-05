import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class FacheDataBase(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        assert type(self.data[index]) is dict and 'sentence_pair' in self.data[index] and 'label' in self.data[index], 'Invalid data format'

        data = self.data[index]['sentence_pair']
        label = self.data[index]['label']
        assert type(data) is tuple and len(data) == 2, 'Invalid data format'

        embeds = [
            self.tokenizer(
                sent,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_token_type_ids=True
            ) for sent in data
        ]
        return {
            'embeds': tuple(embeds),
            'label': label
        }

    def collate_fn(self, batch):
        return {
            'input1': {
                'input_ids': torch.tensor([x['embeds'][0]['input_ids'] for x in batch], dtype=torch.long, device=self.device),
                'token_type_ids': torch.tensor([x['embeds'][0]['token_type_ids'] for x in batch], dtype=torch.long, device=self.device),
                'attention_mask': torch.tensor([x['embeds'][0]['attention_mask'] for x in batch], dtype=torch.long, device=self.device)
            },
            'input2': {
                'input_ids': torch.tensor([x['embeds'][1]['input_ids'] for x in batch], dtype=torch.long, device=self.device),
                'token_type_ids': torch.tensor([x['embeds'][1]['token_type_ids'] for x in batch], dtype=torch.long, device=self.device),
                'attention_mask': torch.tensor([x['embeds'][1]['attention_mask'] for x in batch], dtype=torch.long, device=self.device)
            },
            'label': torch.tensor([x['label'] for x in batch], dtype=torch.long, device=self.device)
        }