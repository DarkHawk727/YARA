"""
Subsystem 3- embedding

Embedding extracted keyword using BERT
Note: No API involved. Local.

download: 
PyTorch- BERT is Implemented in PyTorch. 

side note:
To compare performance, check with Word2Vec by tensorflow.
About BERT-  context-dependent embedding, "Apple" vs Apple in "Apple is red" differs.
So pass a list of string instead of one-by-one approach.
But this code makes a demo approach for one-by-one, but pass-in argument can pass like
[' one chunk'] other than ['one', 'chunk']

"""

from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class BERTKeywordEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
    def get_embeddings(self, keywords):
        keyword_embeddings = []

        for keyword in keywords:
            encoded = self.tokenizer(keyword, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**encoded)
            keyword_embedding = outputs.last_hidden_state[:, 0, :]
            keyword_embeddings.append(keyword_embedding)

        return torch.cat(keyword_embeddings, dim=0)

#example

keyword_embedder = BERTKeywordEmbedder()

#keywords = ['python', 'machine learning', 'natural language processing']
keywords = ['python, machine learning, natural language processing']
#embeddings = keyword_embedder.get_embeddings(keywords)
embeddings = keyword_embedder.get_embeddings(keywords)
embedding_np = np.array(embeddings)
np.set_printoptions(threshold=np.inf)  # This will print the entire array
print(embedding_np)

  
