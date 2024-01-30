import os
from langchain.text_splitter import TokenTextSplitter
from typing import List


def split_text_file(file_path: str, splitter: TokenTextSplitter) -> List[str]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()
        chunks = splitter.split_text(text=content)
    return chunks


def process_directory(input_dir: str, output_dir: str) -> None:
    splitter = TokenTextSplitter()

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path):
            print(f"Processing {file_path}...")
            chunks = split_text_file(file_path, splitter)

            for i, chunk in enumerate(chunks):
                output_file = os.path.join(output_dir, f"{filename}_chunk_{i}.txt")
                with open(output_file, "w", encoding="utf-8") as out_file:
                    out_file.write(chunk)
            print(f"Processed {len(chunks)} chunks for {filename}")


if __name__ == "__main__":
    process_directory(
        input_dir="C:/Users/Arjun Sarao/HamiltonBot/example_documents",
        output_dir="C:/Users/Arjun Sarao/HamiltonBot/test_chunks",
    )

# Now I need to use BERT or some other model to create the embeddings.