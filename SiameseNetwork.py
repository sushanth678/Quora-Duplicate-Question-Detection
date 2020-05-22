import torch
import torch.nn as nn

class Siemese(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,batch_size):
        super().__init__()
        self.bert = bert
        self.batch_size = batch_size
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            batch_first = True)
        self.input_dim = 4 * hidden_dim * 2 if bidirectional else 1
        self.fc1 = nn.Linear(self.input_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)

    def forward_once(self,x,x_mask):
        with torch.no_grad():
            embedded = self.bert(x,x_mask)[0]
            #embedding shape = (batch_size, sentence_length, 768)

        # hidden_size = num_layers * num_directions, batch, hidden_size
        _, hidden = self.rnn(embedded)
        
        # output_dim = (batch_size, 2 * hidden_dim)
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        return output
        
    def forward(self,x,y,x_mask,y_mask):
        # x,y,x_mask,y_mask size = (batch_size, sentence_length)
        r1 = self.forward_once(x,x_mask)
        r2 = self.forward_once(y,y_mask)
    
        absolute = torch.abs(r1 - r2)
        product = r1 * r2

        features = torch.cat((r1,absolute, r2, product),dim = 1)

        fc1 = self.fc1(features)
        out = self.fc2(fc1)
        return out.squeeze(1)