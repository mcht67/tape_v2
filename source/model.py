from torch import nn
import tensorflow as tf
from tensorflow.keras import layers, models

class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_size: int):
        super(Conv1DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # [B, 16, L/2]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, L/4]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 16, L/2]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),   # [B, 1, L]
            nn.Tanh(),  # since sine wave output is in [-1, 1]
        )

    def forward(self, x):
        # x: [B, 1, L]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Conv1DAutoencoder_test(nn.Module):
    def __init__(self, input_size: int):
        super(Conv1DAutoencoder_test, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),  # [B, 16, L/2]
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),  # [B, 32, L/4]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 16, L/2]
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1),   # [B, 1, L]
            nn.Tanh(),  # since sine wave output is in [-1, 1]
        )

    def forward(self, x):
        # x: [B, 1, L]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class SimpleMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_units=[512, 256], dropout_rate=0.3):
        super().__init__()
        
        # Build the sequential stack inside the model
        self.net = models.Sequential()
        self.net.add(layers.Input(shape=input_dim))

        # Flatten spatial / multi-dim input -> (batch_size, num_features)
        self.net.add(layers.Flatten())
        
        # Add hidden layers + dropout after first
        for i, units in enumerate(hidden_units):
            self.net.add(layers.Dense(units, activation='relu'))
            if i == 0:
                self.net.add(layers.Dropout(dropout_rate))
        
        # Output layer
        self.net.add(layers.Dense(1))  # Regression output

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)
    
class ResidualMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_units=512, dropout_rate=0.3):
        super().__init__()
        # Remove this line - InputLayer not needed in functional models
        # self.input_layer = layers.InputLayer(input_shape=input_dim)
        
        # First block
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        # Residual block
        self.dense_res1 = layers.Dense(hidden_units, activation='relu')
        self.dense_res2 = layers.Dense(hidden_units, activation='relu')
        # Output
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=False):
        # Start directly with inputs - no input_layer call
        x = self.dense1(inputs)  # ← Use inputs directly
        x = self.dropout(x, training=training)
        # Residual connection
        res = self.dense_res1(x)
        res = self.dense_res2(res)
        x = layers.add([x, res])
        return self.output_layer(x)
    
class Simple1DCNN(tf.keras.Model):
    def __init__(self, input_dim, filters=64, kernel_size=3, dropout_rate=0.3):
        super().__init__()
        self.reshape = layers.Reshape((input_dim[0], 1))
        self.conv1 = layers.Conv1D(filters, kernel_size, activation='relu')
        self.conv2 = layers.Conv1D(filters, kernel_size, activation='relu')
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)
    
class TemporalCNNMLP(tf.keras.Model):
    def __init__(
        self,
        input_dim=(16, 4, 1536),
        conv_channels=(512, 256),
        mlp_units=(256, 128),
        dropout_rate=0.3,
    ):
        super().__init__()

        self.net = models.Sequential()
        self.net.add(layers.Input(shape=input_dim))

        # --------------------------------------------------
        # 1. Pool over frequency axis
        # (batch, time, freq, channels) -> (batch, time, channels)
        # --------------------------------------------------
        self.net.add(layers.Lambda(lambda x: tf.reduce_mean(x, axis=2)))

        # --------------------------------------------------
        # 2. Temporal convolution stack
        # --------------------------------------------------
        for i, channels in enumerate(conv_channels):
            self.net.add(
                layers.Conv1D(
                    filters=channels,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                )
            )
            self.net.add(layers.BatchNormalization())
            if i == 0:
                self.net.add(layers.Dropout(dropout_rate))

        # --------------------------------------------------
        # 3. Global temporal pooling
        # (batch, time, channels) -> (batch, channels)
        # --------------------------------------------------
        self.net.add(layers.GlobalAveragePooling1D())

        # --------------------------------------------------
        # 4. MLP head
        # --------------------------------------------------
        for i, units in enumerate(mlp_units):
            self.net.add(layers.Dense(units, activation="relu"))
            if i == 0:
                self.net.add(layers.Dropout(dropout_rate))

        # Output layer
        self.net.add(layers.Dense(1))  # regression

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)
    
class TemporalMaxPoolCNNMLP(tf.keras.Model):
    def __init__(
        self,
        input_dim=(16, 4, 1536),
        conv_channels=(512, 256),
        mlp_units=(256, 128),
        dropout_rate=0.3,
    ):
        super().__init__()

        self.net = models.Sequential()
        self.net.add(layers.Input(shape=input_dim))

        # --------------------------------------------------
        # 1. Pool over frequency axis
        # (batch, time, freq, channels) -> (batch, time, channels)
        # --------------------------------------------------
        self.net.add(layers.Lambda(lambda x: tf.reduce_max(x, axis=2)))

        # --------------------------------------------------
        # 2. Temporal convolution stack
        # --------------------------------------------------
        for i, channels in enumerate(conv_channels):
            self.net.add(
                layers.Conv1D(
                    filters=channels,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                )
            )
            self.net.add(layers.BatchNormalization())
            if i == 0:
                self.net.add(layers.Dropout(dropout_rate))

        # --------------------------------------------------
        # 3. Global temporal pooling
        # (batch, time, channels) -> (batch, channels)
        # --------------------------------------------------
        self.net.add(layers.GlobalMaxPooling1D())

        # --------------------------------------------------
        # 4. MLP head
        # --------------------------------------------------
        for i, units in enumerate(mlp_units):
            self.net.add(layers.Dense(units, activation="relu"))
            if i == 0:
                self.net.add(layers.Dropout(dropout_rate))

        # Output layer
        self.net.add(layers.Dense(1))  # regression

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)
    
class TemporalCNNMultiTask(tf.keras.Model):
    def __init__(
        self,
        input_dim=(16, 4, 1536),
        conv_channels=(512, 256),
        dropout_rate=0.3,
    ):
        super().__init__()

        # -----------------------
        # Shared encoder
        # -----------------------
        self.encoder = models.Sequential([
            layers.Input(shape=input_dim),

            # Pool over frequency
            layers.Lambda(lambda x: tf.reduce_mean(x, axis=2)),  # (B, 16, 1536)

            layers.Conv1D(conv_channels[0], kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            layers.Conv1D(conv_channels[1], kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
        ])

        # -----------------------
        # Event detection head
        # -----------------------
        self.event_head = models.Sequential([
            layers.Conv1D(1, kernel_size=1),  # per-time-step logits
        ])

        # -----------------------
        # Counting head
        # -----------------------
        self.count_head = models.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1),  # regression
        ])

    def call(self, inputs, training=False):
        features = self.encoder(inputs, training=training)

        # Event detection (per time step)
        event_logits = self.event_head(features, training=training)
        event_logits = tf.squeeze(event_logits, axis=-1)  # (B, 16)

        # Counting
        count = self.count_head(features, training=training)

        return {
            "polyphony_degree": count,
            "perch2_event_logits": event_logits,
            
        }
    
class TemporalCNNMultiTask_v2(tf.keras.Model):
    def __init__(
        self,
        input_dim=(16, 4, 1536),
        conv_channels=(512, 256),
        dropout_rate=0.3,
        enable_frame_polyphony=True,  # <-- optional switch
    ):
        super().__init__()
        self.enable_frame_polyphony = enable_frame_polyphony

        # -----------------------
        # Shared encoder
        # -----------------------
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_dim),

            # Pool over frequency
            tf.keras.layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=2)
            ),  # (B, 16, 1536)

            tf.keras.layers.Conv1D(
                conv_channels[0], 3, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),

            tf.keras.layers.Conv1D(
                conv_channels[1], 3, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
        ])

        # -----------------------
        # Event detection head
        # -----------------------
        self.event_head = tf.keras.Sequential([
            tf.keras.layers.Conv1D(1, kernel_size=1),
        ])

        # -----------------------
        # Frame-wise polyphony head
        # -----------------------
        if self.enable_frame_polyphony:
            self.frame_polyphony_head = tf.keras.Sequential([
                tf.keras.layers.Conv1D(1, kernel_size=1),
            ])

        # -----------------------
        # Segment polyphony head
        # -----------------------
        self.count_head = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1),
        ])

    def call(self, inputs, training=False):
        features = self.encoder(inputs, training=training)

        outputs = {}

        # Event detection (B, 16)
        event_logits = self.event_head(features, training=training)
        outputs["perch2_event_logits"] = tf.squeeze(event_logits, axis=-1)

        # Frame-wise polyphony (B, 16)
        if self.enable_frame_polyphony:
            frame_poly = self.frame_polyphony_head(features, training=training)
            outputs["frame_polyphony"] = tf.squeeze(frame_poly, axis=-1)

        # Segment-level polyphony (B, 1)
        outputs["polyphony_degree"] = self.count_head(features, training=training)

        return outputs




