import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    """Learned positional embeddings for sequential network flow data."""
    def __init__(self, max_seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=max_seq_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class CustomTransformerEncoder(layers.Layer):
    """Custom Transformer Encoder block utilizing Multi-Head Attention."""
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Attention block with residual connection
        attn_output = self.mha(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward block with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class FlowTransformer(keras.Model):
    """Self-Supervised Masked Flow Transformer for anomaly detection."""
    def __init__(self, feature_dim, d_model=64, num_heads=4, num_layers=3, dff=128, max_seq_len=256, mask_prob=0.15):
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.input_proj = layers.Dense(d_model)
        self.pos_encoding = PositionalEmbedding(max_seq_len, d_model)
        self.encoder_layers = [
            CustomTransformerEncoder(d_model=d_model, num_heads=num_heads, dff=dff, dropout=0.1)
            for _ in range(num_layers)
        ]
        self.reconstruction_head = layers.Dense(feature_dim, activation='linear')

    def call(self, inputs, training=False):
        x = self.input_proj(inputs)  
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return self.reconstruction_head(x)

    def mask_flows(self, x):
        """Dynamically applies a random binary mask to input sequences."""
        shape = tf.shape(x)
        batch_size, seq_len = shape[0], shape[1]
        mask = tf.random.uniform((batch_size, seq_len)) > self.mask_prob
        mask = tf.cast(mask, tf.float32)
        masked_x = x * tf.expand_dims(mask, -1)
        return masked_x, x, mask

    def train_step(self, data):
        """Custom training logic for self-supervised reconstruction."""
        x = data  
        masked_x, target, mask = self.mask_flows(x)
        with tf.GradientTape() as tape:
            reconstructed = self(masked_x, training=True)
            loss = self.reconstruction_loss(target, reconstructed, mask)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def reconstruction_loss(self, target, pred, mask):
        """Calculates Mean Squared Error strictly on masked positions."""
        diff = tf.square(target - pred)
        diff = tf.reduce_sum(diff, axis=-1)  
        masked_diff = diff * (1 - mask)  # 1 - mask targets the hidden slots
        return tf.reduce_sum(masked_diff) / (tf.reduce_sum(1 - mask) + 1e-8)

    def compute_anomaly_score(self, x):
        """Computes the anomaly score based on maximum reconstruction deviation."""
        reconstructed = self(x, training=False)
        error_per_flow = tf.reduce_mean(tf.square(x - reconstructed), axis=-1)  
        return tf.reduce_max(error_per_flow, axis=-1)  

if __name__ == "__main__":
    # Simulate dummy network telemetry data (1000 windows, seq_len 128, 8 metrics)
    num_samples = 1000
    seq_len = 128
    feature_dim = 8

    normal_windows = np.random.normal(size=(num_samples, seq_len, feature_dim)).astype(np.float32)
    new_window = np.random.normal(size=(1, seq_len, feature_dim)).astype(np.float32)

    # Initialize data pipeline
    dataset = tf.data.Dataset.from_tensor_slices(normal_windows).batch(32).prefetch(tf.data.AUTOTUNE)

    # Instantiate and compile model
    model = FlowTransformer(feature_dim=feature_dim, d_model=64, num_heads=4, num_layers=3)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    # Pre-training
    print("Starting Self-Supervised Pre-training...")
    model.fit(dataset, epochs=3) 

    # Anomaly evaluation
    print("\nEvaluating a new network window...")
    score = model.compute_anomaly_score(new_window)
    print(f"Anomaly Score: {score.numpy()[0]:.4f}")
