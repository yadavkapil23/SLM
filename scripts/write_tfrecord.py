import tensorflow as tf
from tokenizers import ByteLevelBPETokenizer
import os
from tqdm import tqdm

def encode_and_write_tfrecord(corpus_path, tokenizer_path, tfrecord_path, seq_len=128):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    vocab_path = f"{tokenizer_path}/vocab.json"
    merges_path = f"{tokenizer_path}/merges.txt"
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"Tokenizer files not found in: {tokenizer_path}")
    
    tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

    num_written = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Tokenizing"):
                ids = tokenizer.encode(line.strip()).ids
                for i in range(0, len(ids) - seq_len):
                    input_ids = ids[i:i+seq_len]
                    label_ids = ids[i+1:i+seq_len+1]
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
                        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=label_ids))
                    }))
                    writer.write(example.SerializeToString())
                    num_written += 1

    print(f"✅ TFRecord written: {tfrecord_path}")
    print(f"✏️ Total examples: {num_written}")

if __name__ == "__main__":
    encode_and_write_tfrecord("data/corpus.txt", "tokenizer", "data/dataset.tfrecord")
