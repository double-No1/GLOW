from transformers import BertModel
import torch.nn as nn

class BertEmbeddingModel(BertModel):
    def __init__(self, config, output_dim=None):
        super().__init__(config)
        self.output_dim = output_dim
        if output_dim is not None:
            self.linear_transform = nn.Linear(config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        if self.output_dim is not None:
            embeddings = self.linear_transform(embeddings)
        return embeddings






