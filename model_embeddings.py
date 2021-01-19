#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"



class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_char_size = 50
        self.embed_size = embed_size
        self.char_embed=nn.Embedding(num_embeddings=len(vocab.char2id),embedding_dim=self.embed_char_size,padding_idx=pad_token_idx)
        self.cnn=CNN(input_channel=self.embed_char_size,output_channel=self.embed_size)
        self.highway = Highway(input_size=self.embed_size)
        self.Dropout=nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        #output = self.embeddings(input)
        #return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        print(input)
        sentence_length, batch_size, max_word_length=input.shape
        embed=self.char_embed(input) #sentence_length, batch_size, max_word_length,embed_char_size
        embed_reshaped=embed.reshape(embed.size(0)*embed.size(1),embed.size(2),embed.size(3)).permute(0,2,1)#sentence_length*batch_size,embed_char_size,max_word_length
        conv_out=self.cnn(embed_reshaped)
        highway=self.highway(conv_out)
        word_embed=self.Dropout(highway)
        out_word_embed=word_embed.reshape(sentence_length,batch_size,word_embed.size(1))
        return out_word_embed
        ### END YOUR CODE

