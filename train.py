import tensorflow as tf
from model.tiny_transformer import TinyTransformer

def parse_tfrecord(example_proto, seq_len):
    feature_description = {
        "input_ids": tf.io.FixedLenFeature([seq_len], tf.int64),
        "labels": tf.io.FixedLenFeature([seq_len], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed["input_ids"], parsed["labels"]

def load_dataset(tfrecord_path, seq_len=128, batch_size=32, validation_split=0.1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: parse_tfrecord(x, seq_len))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Split into train and validation
    total_size = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    train_size = int(total_size * (1 - validation_split))
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    return train_dataset, val_dataset

def train():
    vocab_size = 30000
    seq_len = 128
    embed_dim = 128
    num_heads = 2
    ff_dim = 512
    batch_size = 32
    epochs = 5

    model = TinyTransformer(vocab_size, embed_dim, num_heads, ff_dim)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    train_dataset, val_dataset = load_dataset("data/dataset.tfrecord", seq_len, batch_size)
    
    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    
    model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    model.save("trained_slm_model")

if __name__ == "__main__":
    train()
