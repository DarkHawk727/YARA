""""

makes a .txt of keywords, separated by \n to represent passages
into BERT embeddings as DQN state format. 

BERT recommend using 768 dimensions for DQN embeddings

"""

from transformers import AutoTokenizer, AutoModel
import torch
import argparse

class BertEmbed_s:
    def __init__(self, model_name, embedding_dim):
        # Initialize BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.embedding_dim = embedding_dim

    def read_file(self, file_path):
        passages = []
        with open(file_path, 'r') as file:
            passage = []
            for line in file:
                line = line.strip()
                if line:
                    passage.extend(line.split(', '))
                else:
                    passages.append(passage)
                    passage = []
            if passage:
                passages.append(passage)
        return passages

    def generate_embeddings(self, passages):
        embeddings = []
        for passage in passages:
            text = ' '.join(passage)
            tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                output = self.model(**tokens)
                last_hidden_state = output.hidden_states[-1]
                # Reduce the embedding dimension using linear projection
                reduced_embedding = torch.mean(last_hidden_state, dim=1)
                reduced_embedding = reduced_embedding[:, :self.embedding_dim]  # Trim to desired dimension
                embeddings.append(reduced_embedding.numpy())
        return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, help="Path to the .txt file containing passages with keywords")
    args = parser.parse_args()

    # Initialize the BertEmbed_s class with the desired embedding dimension (e.g., 128)
    keyword_embedding = BertEmbed_s("bert-base-uncased", embedding_dim=128) #### 756 for later

    # Read the passages from the specified text file
    passages = keyword_embedding.read_file(args.file_path)

    # Generate BERT embeddings with the reduced dimension for each passage
    embeddings = keyword_embedding.generate_embeddings(passages)

    # Print the embeddings for each passage
    for i, embedding in enumerate(embeddings):
        print(f"Passage {i + 1} Embedding: {embedding}")

if __name__ == "__main__":
    main()


