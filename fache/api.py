import os
import torch
import spacy

from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from omegaconf import DictConfig
from transformers import AutoTokenizer
from importlib import import_module

from fache.model import FacheModel

class FacheRequest(BaseModel):
    claim: str
    para: str

class FacheAPI(FastAPI):
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.dir)
        
        self.model = FacheModel.load_from_checkpoint(
            os.path.join(cfg.model.dir, 'best.ckpt'),
            model_arch=cfg.model.arch,
            classifier_hidden_size=cfg.model.classifier_hidden_size,
        )
        self.model.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(torch.device(self.device))

        self.dataset = import_module(
            "fache.tasks.{}.dataset".format(cfg.run.task)
        ).FacheData(
            tokenizer=self.tokenizer, 
            max_len=cfg.data.max_len, 
            split='server', 
            data_dir='/'
        )

        self.task = cfg.run.task
        self.cfg = cfg
        self.nlp = spacy.load('ja_ginza_electra')

        self.post("/api/predict")(self.predict)
    
    def predict(self, req: FacheRequest) -> List[Dict[str, Any]]:
        embed2 = self.tokenizer(
            req.para, 
            padding=False,
            truncation=True,
            max_length=self.dataset.max_len,
            return_token_type_ids=True
        )
        model_input2 = {
            'input_ids': torch.tensor([embed2['input_ids']], dtype=torch.long, device=self.device),
            'token_type_ids': torch.tensor([embed2['token_type_ids']], dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor([embed2['attention_mask']], dtype=torch.long, device=self.device)
        }

        output: List[Dict[str, Any]] = []
        doc = self.nlp(req.claim)
        for sent in doc.sents:
            embed1 = self.tokenizer(
                sent.text, 
                padding=False,
                truncation=True,
                max_length=self.dataset.max_len,
                return_token_type_ids=True
            )
            model_input1 = {
                'input_ids': torch.tensor([embed1['input_ids']], dtype=torch.long, device=self.device),
                'token_type_ids': torch.tensor([embed1['token_type_ids']], dtype=torch.long, device=self.device),
                'attention_mask': torch.tensor([embed1['attention_mask']], dtype=torch.long, device=self.device)
            }

            with torch.no_grad():
                pred = self.model(model_input1, model_input2)

            label = torch.argmax(pred, dim=1).item()
            score = float(torch.max(torch.nn.functional.softmax(pred, dim=1)))

            output.append({
                "sent": sent.text,
                "label": label,
                "score": score,
                # "config": self.cfg
            })

        return output