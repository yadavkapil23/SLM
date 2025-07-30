from tokenizers import ByteLevelBPETokenizer
import os

def train_tokenizer(corpus_path, save_path, vocab_size=30000):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train([corpus_path], vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_model(save_path)

if __name__ == "__main__":
    train_tokenizer("data/corpus.txt", "tokenizer/")
