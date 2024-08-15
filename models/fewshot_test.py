import torch
import torch.nn as nn

class DFusionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(DFusionAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = dropout

        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)

        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs1, inputs2):
        batch_size = inputs1.size(0)

        query1 = self.query1(inputs1).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key1 = self.key1(inputs1).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value1 = self.value1(inputs1).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        query2 = self.query2(inputs2).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key2 = self.key2(inputs2).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value2 = self.value2(inputs2).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        attention_scores = torch.matmul(query1, key1.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attention_probs = self.softmax(attention_scores)
        attention_probs = nn.Dropout(self.dropout)(attention_probs)

        context1 = torch.matmul(attention_probs, value1).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        attention_scores = torch.matmul(query2, key2.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attention_probs = self.softmax(attention_scores)
        attention_probs = nn.Dropout(self.dropout)(attention_probs)

        context2 = torch.matmul(attention_probs, value2).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        fused_context = torch.cat((context1, context2), dim=-1)
        fused_context = nn.Linear(self.hidden_size*2, self.hidden_size)(fused_context)

        return fused_context
