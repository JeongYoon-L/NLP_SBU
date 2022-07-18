"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math 

import numpy as np
import torch
import torch.nn as nn

sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func, counts):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func
        self.counts = counts

    def forward(self, center_word, context_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))
    
    def negative_log_likelihood_loss(self, center_word, context_word):
        ### TODO(students): start
        
        vc = self.center_embeddings(center_word)
        uk = self.context_embeddings.weight
        u0 = self.context_embeddings(context_word)
        
        first = (u0 * vc).sum(dim =1) * -1 #u0_T vc  = [batch size *1]

        second = torch.matmul(uk,torch.t(vc))
        second = torch.logsumexp(second,dim = 0) # [1*batch size]
        
        nll_loss = first + torch.t(second)
        nll_loss = nll_loss.mean()
        loss = nll_loss
        
        ### TODO(students): end

        return loss
    
    def negative_sampling(self, center_word, context_word):
        
        ### TODO(students): start
        
        k = 1
        vc = self.center_embeddings(center_word)
        u0 = self.context_embeddings(context_word)
        
        k_samples = (torch.tensor(self.counts).cuda())**(0.75)
        P_w = (k_samples/torch.sum(k_samples))
        k_index = torch.multinomial(P_w, k, replacement = True)
        uk = self.context_embeddings(k_index)
        
        ng_second = torch.sum(torch.log(sigmoid(torch.matmul(-uk,torch.transpose(vc,0,1)))),dim = 0) # [batch size *128]
        
        ng_first = u0* vc
        ng_first_sum = (-1)* torch.log(sigmoid(torch.sum(ng_first,dim = 1)))
        
        loss = ng_first_sum - ng_second
        loss = loss.mean()
        
        ### TODO(students): end
        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])            
            
            