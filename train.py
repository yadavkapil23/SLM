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
    # Load the dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: parse_tfrecord(x, seq_len))
    dataset = dataset.shuffle(1000)
    
    # Use a more efficient approach for splitting
    # Calculate approximate split based on file size or use a fixed split
    dataset_size = 29909128  # Known size from the README
    train_size = int(dataset_size * (1 - validation_split))
    
    # Split the dataset
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Apply batching and prefetching
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def train():
    # Enable mixed precision for T4
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # T4-specific optimizations
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    tf.config.optimizer.set_experimental_options({"layout_optimizer": True})
    
    vocab_size = 30000
    seq_len = 128
    embed_dim = 128
    num_heads = 2
    ff_dim = 512
    batch_size = 128  # Increased for faster training
    epochs = 3  # Reduced epochs - let early stopping handle it
    
    # Create model with dropout for overfitting prevention
    model = TinyTransformer(vocab_size, embed_dim, num_heads, ff_dim)
    
    # Compile with stronger regularization
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=2e-4,  # Slightly higher for faster convergence
            weight_decay=1e-4,   # Stronger L2 regularization
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        metrics=['accuracy'],
        jit_compile=True  # Enable XLA for model compilation
    )

    train_dataset, val_dataset = load_dataset("data/dataset.tfrecord", seq_len, batch_size)
    
    # Calculate steps per epoch for better monitoring
    total_train_examples = 26918215  # ~27M from README
    steps_per_epoch = total_train_examples // batch_size
    
    print(f"Training on Tesla T4 GPU")
    print(f"Dataset: {total_train_examples:,} training examples")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Estimated time per epoch: ~1-1.5 hours on T4")
    print(f"Total training time: ~3-4.5 hours")
    print(f"Overfitting prevention: Dropout + L2 + Early Stopping")
    
    # Enhanced callbacks for overfitting prevention and faster training
    callbacks = [
        # Early stopping with more aggressive settings
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,  # Reduced patience for faster stopping
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,  # Reduced patience
            min_lr=1e-6,
            verbose=1
        ),
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard for monitoring
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        ),
        # Custom callback to reduce training time
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1} completed - Val Loss: {logs.get('val_loss', 0):.4f}")
        )
    ]
    
    print("Training model...")
    model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    model.save("trained_slm_model")

if __name__ == "__main__":
    train()
