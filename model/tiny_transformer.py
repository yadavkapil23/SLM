import tensorflow as tf

class TinyTransformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, max_seq_len=512):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(max_seq_len, embed_dim)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices; cos to odd indices
        sin_part = tf.sin(angle_rads[:, 0::2])
        cos_part = tf.cos(angle_rads[:, 1::2])
        
        # Create the full positional encoding by interleaving sin and cos
        pos_encoding = tf.zeros((position, d_model), dtype=tf.float32)
        
        # Use scatter_nd to place sin values at even indices
        even_indices = tf.stack([
            tf.repeat(tf.range(position), d_model//2),
            tf.tile(tf.range(0, d_model, 2), [position])
        ], axis=1)
        
        # Use scatter_nd to place cos values at odd indices  
        odd_indices = tf.stack([
            tf.repeat(tf.range(position), d_model//2),
            tf.tile(tf.range(1, d_model, 2), [position])
        ], axis=1)
        
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding, even_indices, tf.reshape(sin_part, [-1])
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding, odd_indices, tf.reshape(cos_part, [-1])
        )
        
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x_embed = self.embedding(x)
        x_embed += self.pos_encoding[:, :seq_len, :]
        
        attn_out = self.attn(x_embed, x_embed)
        out1 = self.norm1(x_embed + attn_out)
        ffn_out = self.ffn(out1)
        logits = self.output_layer(self.norm2(out1 + ffn_out))
        return logits
