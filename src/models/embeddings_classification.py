import torch
from torch import nn
import math

class MLP(nn.Module):
    def __init__(self, emb_dim, num_class, num_bottleneck = 100, num_bottleneck1 = 50, dropout = 0.5):
        super(MLP, self).__init__()
        self.num_class = num_class
        self.emb_dim = emb_dim
        self.num_bottleneck =num_bottleneck
        self.num_bottleneck1 = num_bottleneck1
        self.dropout = dropout
        self.classifier = self.fc_fusion()

        self._initialize_weights()

        
    def fc_fusion(self):
        classifier = nn.Sequential(
                nn.Linear(self.emb_dim, self.num_bottleneck),
                nn.BatchNorm1d(self.num_bottleneck),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.num_bottleneck, self.num_bottleneck1),
                nn.BatchNorm1d(self.num_bottleneck1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.num_bottleneck1, self.num_class),
                )
        return classifier
    
    def forward(self, input):
        output = self.classifier(input)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class LSTM(nn.Module):
    def __init__(self, emb_dim, num_class, num_bottleneck = 100, hidden_dim = 128, num_layers = 1, dropout = 0.5):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, num_bottleneck)
        self.fc2 = nn.Linear(num_bottleneck, num_class)
    
    def forward(self, x):
        # Add an extra dimension to the input tensor to match the LSTM input shape
        x = x.unsqueeze(1)  # Now x shape is (batch_size, 1, input_size)
        #logger.info(f"Input Shape: {x.shape}")
        #logger.info(f"x.size(1) = {x.size(1)}")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendere l'output del ultimo timestep
        out = self.dropout(out)
        embeddings = self.fc1(out)
        out = self.fc2(embeddings)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class Transformer(nn.Module):
    def __init__(self, emb_dim, num_classes, num_heads=4, num_layers=1, d_ff=100, max_seq_length=1,  num_bottleneck=100, dropout=0.1):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(emb_dim, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(emb_dim, num_bottleneck)
        self.fc2 = nn.Linear(num_bottleneck, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x.shape: (batch_size, num_clips, d_model)
        x = x.unsqueeze(1)  # Now x shape is (batch_size, 1, input_size)
        x = self.positional_encoding(x)
        
        # Permuta per adattare alla forma attesa dal Transformer Encoder
        x = x.permute(1, 0, 2)  # (num_clips, batch_size, d_model)
        
        # Passa attraverso il Transformer Encoder
        transformer_out = self.transformer_encoder(x)
        
        # Media le uscite del Transformer per ogni clip
        transformer_out = transformer_out.mean(dim=0)  # (batch_size, d_model)
        
        # Passa attraverso il livello finale di classificazione
        output = self.fc(self.dropout(transformer_out))  # (batch_size, num_classes)

        output = self.fc2(output)
        
        return output