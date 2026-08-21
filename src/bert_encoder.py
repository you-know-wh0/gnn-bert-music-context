import torch
import torch.nn as nn
from transformers import BertModel

class BertTagClassifier(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-uncased', freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]     # [CLS] token → shape (batch, 768)
        logits = self.classifier(cls)
        if return_embedding:
            return logits, cls
        return logits