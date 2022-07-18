'''
author: Sounak Mondal
'''

# std lib imports
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.hidden_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for i in range(num_layers-1)])
        self.dropout = nn.Dropout(dropout)
        self.last_layer = nn.Linear(input_dim, input_dim)
        
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        # print(vector_sequence.shape) #torch.Size([64, 209, 50]) , (batch_size, max_tokens_num, embedding_dim)
        # print(sequence_mask.shape) #torch.Size([64, 209]), (batch_size, max_tokens_num)
        
        #dropout
        sequence_mask = self.dropout(sequence_mask)
        #padding
        sequence_mask = sequence_mask.unsqueeze(-1) # make it to 3D Tensor, torch.Size([64, 209,1]), (batch_size, max_tokens_num)
        after_padding = sequence_mask * vector_sequence 
        after_padding = after_padding.mean(dim = 1) #averaging. torch.Size([64, 209]), (batch_size, max_tokens_num)
        vector_sequence = after_padding
        
        self.for_layer_representations = []
        for layer in self.hidden_layer:
            vector_sequence = layer(vector_sequence)
            combined_vector = vector_sequence
            self.for_layer_representations.append(vector_sequence)
            nn.ReLU(vector_sequence)
            
        vector_sequence = self.last_layer(vector_sequence)
        combined_vector = vector_sequence
        self.for_layer_representations.append(vector_sequence)
        
        layer_representations = self.for_layer_representations
        layer_representations = torch.stack(layer_representations, dim=1)
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}



class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.linears = nn.GRU(self._input_dim,self._input_dim,num_layers, batch_first = True)
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start

        new_vector_sequence = torch.nn.utils.rnn.pack_padded_sequence(vector_sequence, torch.sum(sequence_mask, dim=1).long().cpu(), batch_first=True, enforce_sorted=False)
        
        aa ,layer_output = self.linears(new_vector_sequence)
        # print(layer_output.shape) #[4,64,50]
        
        layer_representations = layer_output.permute(1,0,2) #64,4,50
        # layer_representations = layer_representations.tolist()
        combined_vector = layer_output[3][:][:] #64,50
        
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}

