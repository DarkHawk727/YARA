"""
Code purpose:

Use tokenizer to predict embedding token counts, which is not free. 
No API token required.

requirement: pip/ pip3
pip install tiktoken
pip install openai 

Intro:
token counting code

API source :#encoding= tiktoken.encoding_for_model("gpt-3.5-turbo")
https://github.com/openai/tiktoken

cookbook example
https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb 


script limitations:
python string does not specify max. amount of words but may still be limited due to memory.

"""

import tiktoken 



class Token_count:

    def __init__(self, optional_input=None):
        #pass
        #self.input=input
        #self.output=output(input, "cl100k_base")
        self.optional_input = optional_input

    def num_tokens_from_string(self, string: str, encoding_name= "cl100k_base"): # cl100k_base aka 2ed gen, first tokenize then emdedded, we use  KNOWLEDGE CUTOFF Sep 2021  https://platform.openai.com/docs/guides/embeddings/what-are-embeddings 

        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        #debug
        print(f' {num_tokens}: {string} \n')
        
        return num_tokens


# i.e. print (num_tokens_from_string("tiktoken is great!", "cl100k_base"))
 

