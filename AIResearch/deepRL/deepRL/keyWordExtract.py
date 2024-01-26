"""
Subsystem 3- Parse keyword
problem format:
 %%% keyword1, keyword2,... keywordn %%% content 
solution: (unrestricted).

Side note:
during training- prompt trainset to follow the format above.

during deployement- two GPT prompt: 1.keyword tokenizer 2. response generizer 

"""

import re
#from BERTKeywordEmbedder import BERTKeywordEmbedder, np #debug


def extract_keywords_and_content(text, flatten_keywords=False):
    start_index = text.find("%%%") + 3
    end_index = text.find("%%%", start_index)
    keywords = text[start_index:end_index].split(',')
    
    
    if flatten_keywords:
        result_string = ','.join(keywords) # flatten: 'keyword1, keyword 2, ..n' for BERT embedding
        return result_string
    
    return keywords

""" 
#debug 
# Example usage
input_string = '%%%keyword1,keyword2,keyword3%%%This is the content.'
keywords = extract_keywords_and_content(input_string,True)

keyword_embedder = BERTKeywordEmbedder()
embeddings = keyword_embedder.get_embeddings(keywords)
embedding_np = np.array(embeddings)
np.set_printoptions(threshold=np.inf)  # This will print the entire array
print(embedding_np)

if keywords is not None:
    print('Keywords:', keywords)
    #print('Content:', content)
else:
    print('No keywords found.')

"""
