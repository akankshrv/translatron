import torch 
import torch.nn as nn
import math 


#InputEmbeddings is a map between number and vector of size 512
#d_model = 512
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int, vocab_size:int)->None:
        super().__init__
        self.d_model = d_model
        self.vocab_size =vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)#lookup table
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositonalEncoding(nn.Module):
      
    def __init__(self,d_model:int, seq_len:int, dropout:float)->None:
        super().__init__
        self.d_model = d_model
        self.deq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matrix of shape(seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)
        #vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #numerator

        div_term = torch.exp(torch.arrange(0,d_model,2).float() * (-math.log(10000.0)/ d_model))#denominator

        #Apply sin to even position 
        pe[:,0::2]= torch.size(position * div_term)
        #Apply cos to odd position
        pe[:,1::2] = torch.size(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self,x):
        x = x +(self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps:float = 10**-6)-> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.beta = nn.Parameter(torch.zeros(1)) #Added

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
