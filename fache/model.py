import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import AutoConfig, AutoModel
from transformers.trainer_pt_utils import get_parameter_names
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, label_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, label_size)
    
    def forward(self, input1, input2):
        input = torch.cat([input1, input2], dim=1)
        return self.linear2(self.relu(self.linear1(input)))

class FacheModel(pl.LightningModule):
    def __init__(self, model_arch, classifier_hidden_size):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_arch)
        self.model_cfg = AutoConfig.from_pretrained(model_arch)
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = Classifier(
            2*self.model_cfg.hidden_size,
            classifier_hidden_size,
            2
        )
        self.dropout = nn.Dropout(
            self.model_cfg.classifier_dropout \
            if self.model_cfg.classifier_dropout is not None else \
            self.model_cfg.hidden_dropout_prob
        )

    def forward(self, input1, input2):
        hiddens = []
        for input in [input1, input2]:
            enc_out = self.encoder(**input)
            hidden = enc_out['last_hidden_state'][:, 0, :]
            hidden = self.dropout(hidden)
            hiddens.append(hidden)
        logit = self.classifier(*hiddens)
        return logit

    def training_step(self, batch, batch_idx):
        logit = self.forward(batch['input1'], batch['input2'])
        targets = batch['label']
        loss = self.criterion(logit, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            logit = self.forward(batch['input1'], batch['input2'])
        targets = batch['label']
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, dim=1)
        return {
            'loss': loss,
            'preds': preds,
            'targets': targets
        }

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        targets = torch.cat([x['targets'] for x in outputs]).detach().cpu().numpy()
        acc = accuracy_score(targets, preds)
        self.log('val_acc', acc)
        logger.info('\n' + classification_report(targets, preds, digits=4) + '\n')

    def configure_optimizers(self):
        decay_params = get_parameter_names(self, [nn.LayerNorm])
        decay_params = [name for name in decay_params if 'bias' not in name]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if n in decay_params],
                'weight_decay': 0.01
            },
            {
                'params': [p for n,p in self.named_parameters() if n not in decay_params],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=3e-5
        )
        return optimizer