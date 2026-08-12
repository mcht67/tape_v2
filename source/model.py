from torch import nn
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable

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
    
# class SimpleMLP(tf.keras.Model):
#     def __init__(self, input_dim, hidden_units=[512, 256], dropout_rate=0.3):
#         super().__init__()
        
#         # Build the sequential stack inside the model
#         self.net = models.Sequential()
#         self.net.add(layers.Input(shape=input_dim))

#         # Flatten spatial / multi-dim input -> (batch_size, num_features)
#         self.net.add(layers.Flatten())
        
#         # Add hidden layers + dropout after first
#         for i, units in enumerate(hidden_units):
#             self.net.add(layers.Dense(units, activation='relu'))
#             if i == 0:
#                 self.net.add(layers.Dropout(dropout_rate))
        
#         # Output layer
#         self.net.add(layers.Dense(1))  # Regression output

#     def call(self, inputs, training=False):
#         return self.net(inputs, training=training)
    
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
            "polyphony": count,
            "perch2_event_logits": event_logits,
            
        }

# @register_keras_serializable(package="model", name="TemporalCNNMultiTask_v2")
# class TemporalCNNMultiTask_v2(tf.keras.Model):
#     def __init__(
#         self,
#         input_dim=(16, 4, 1536),
#         conv_channels=(512, 256),
#         dropout_rate=0.3,
#         enable_segment_polyphony=False,
#         enable_frame_polyphony=False,
#         enable_event_logits=False,
#         **kwargs):
#         super().__init__(**kwargs)
#         self.enable_segment_polyphony = enable_segment_polyphony
#         self.enable_frame_polyphony = enable_frame_polyphony
#         self.enable_event_logits = enable_event_logits

#         # -----------------------
#         # Shared encoder
#         # -----------------------
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=input_dim),

#             # Pool over frequency
#             tf.keras.layers.Lambda(
#                 lambda x: tf.reduce_mean(x, axis=2)
#             ),  # perch_v2_spatial: (B, 16, 1536)

#             tf.keras.layers.Conv1D(
#                 conv_channels[0], 3, padding="same", activation="relu"
#             ),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(dropout_rate),

#             tf.keras.layers.Conv1D(
#                 conv_channels[1], 3, padding="same", activation="relu"
#             ),
#             tf.keras.layers.BatchNormalization(),
#         ])

#         # -----------------------
#         # Event detection head
#         # -----------------------
#         if self.enable_event_logits:
#             self.event_head = tf.keras.Sequential([
#                 tf.keras.layers.Conv1D(1, kernel_size=1),
#             ])

#         # -----------------------
#         # Frame-wise polyphony head
#         # -----------------------
#         if self.enable_frame_polyphony:
#             self.frame_polyphony_head = tf.keras.Sequential([
#                 tf.keras.layers.Conv1D(1, kernel_size=1),
#             ])

#         # -----------------------
#         # Segment polyphony head
#         # -----------------------
#         if self.enable_segment_polyphony:
#             self.segment_polyphony_head = tf.keras.Sequential([
#                 tf.keras.layers.GlobalAveragePooling1D(),
#                 tf.keras.layers.Dense(128, activation="relu"),
#                 tf.keras.layers.Dropout(dropout_rate),
#                 tf.keras.layers.Dense(1),
#             ])

#     def call(self, inputs, training=False):
#         features = self.encoder(inputs, training=training)

#         outputs = {}

#         # Event detection perch_v2_spatial: (B, 16)
#         if self.enable_event_logits:
#             event_logits = self.event_head(features, training=training)
#             outputs["event_logits"] = tf.squeeze(event_logits, axis=-1)

#         # Frame-wise polyphony perch_v2_spatial: (B, 16)
#         if self.enable_frame_polyphony:
#             frame_poly = self.frame_polyphony_head(features, training=training)
#             outputs["framewise_polyphony_reg"] = tf.squeeze(frame_poly, axis=-1)

#         # Segment-level polyphony perch_v2_spatial: (B, 1)
#         if self.enable_segment_polyphony:
#             outputs["polyphony_degree"] = self.segment_polyphony_head(features, training=training)

#         return outputs
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "input_dim": self.input_dim,
#             "conv_channels": self.conv_channels,
#             "dropout_rate": self.dropout_rate,
#             "enable_segment_polyphony": self.enable_segment_polyphony,
#             "enable_frame_polyphony": self.enable_frame_polyphony,
#             "enable_event_logits": self.enable_event_logits,
#         })
#         return config
    
#     def get_loss_config(self):
#         """Return dict defining what losses this model needs."""
#         loss_config = {}
        
#         if self.enable_event_logits:
#             loss_config["event_logits"] = {
#                 "type": tf.keras.losses.BinaryCrossentropy,
#                 "from_logits": True,
#                 "weight": 1.0
#             }
        
#         if self.enable_frame_polyphony:
#             loss_config["framewise_polyphony_reg"] = {
#                 "type": tf.keras.losses.MeanSquaredError,
#                 "weight": 1.0
#             }
        
#         if self.enable_segment_polyphony:
#             loss_config["polyphony_degree"] = {
#                 "type": tf.keras.losses.MeanSquaredError,
#                 "weight": 1.0
#             }
        
#         return loss_config

# @register_keras_serializable(package="model", name="TemporalCNNMultiTask_v2")
# class TemporalCNNMultiTask_v2(tf.keras.Model):
#     def __init__(
#         self,
#         input_dim=(None, 4, 1536),
#         conv_channels=(512, 256),
#         dropout_rate=0.3,
#         objectives=None,
#         **kwargs
#     ):
#         super().__init__(**kwargs)

#         # Store config
#         self.input_dim = input_dim
#         self.conv_channels = conv_channels
#         self.dropout_rate = dropout_rate
#         self.objectives = list(objectives) or []
        
#         # Build encoder
#         self.freq_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=2))
#         self.conv1 = tf.keras.layers.Conv1D(conv_channels[0], 3, padding="same", activation="relu")
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.conv2 = tf.keras.layers.Conv1D(conv_channels[1], 3, padding="same", activation="relu")
#         self.bn2 = tf.keras.layers.BatchNormalization()
        
#         # Build heads based on objectives
#         if "event_logits" in self.objectives:
#             self.event_head = tf.keras.layers.Conv1D(1, kernel_size=1)
        
#         if "framewise_polyphony_reg" in self.objectives:
#             self.frame_polyphony_head = tf.keras.layers.Conv1D(1, kernel_size=1)
        
#         if "polyphony_degree" in self.objectives:
#             self.segment_pool = tf.keras.layers.GlobalAveragePooling1D()
#             self.segment_dense1 = tf.keras.layers.Dense(128, activation="relu")
#             self.segment_dropout = tf.keras.layers.Dropout(dropout_rate)
#             self.segment_dense2 = tf.keras.layers.Dense(1)
    
#     def call(self, inputs, training=False):
#         # Encoder
#         x = self.freq_pool(inputs)
#         x = self.conv1(x)
#         x = self.bn1(x, training=training)
#         x = self.dropout1(x, training=training)
#         x = self.conv2(x)
#         features = self.bn2(x, training=training)
        
#         outputs = {}
        
#         if "event_logits" in self.objectives:
#             event_logits = self.event_head(features, training=training)
#             outputs["event_logits"] = tf.squeeze(event_logits, axis=-1)
        
#         if "framewise_polyphony_reg" in self.objectives:
#             frame_poly = self.frame_polyphony_head(features, training=training)
#             outputs["framewise_polyphony_reg"] = tf.squeeze(frame_poly, axis=-1)
        
#         if "polyphony_degree" in self.objectives:
#             x = self.segment_pool(features)
#             x = self.segment_dense1(x)
#             x = self.segment_dropout(x, training=training)
#             outputs["polyphony_degree"] = self.segment_dense2(x)
        
#         return outputs
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "input_dim": self.input_dim,
#             "conv_channels": self.conv_channels,
#             "dropout_rate": self.dropout_rate,
#             "objectives": self.objectives,
#         })
#         return config
    
# @register_keras_serializable(package="model", name="TemporalCNN")
# class TemporalCNN(tf.keras.Model):
#     def __init__(
#         self,
#         input_dim=(None, 4, 1536),
#         conv_channels=(512, 256),
#         dropout_rate=0.3,
#         objectives_cfg=None,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         # Store config
#         self.input_dim = input_dim
#         self.conv_channels = list(conv_channels)
#         self.dropout_rate = dropout_rate
#         # self.objectives_cfg = {obj_name: ({"num_classes": obj_cfg["num_classes"]} if "num_classes" in obj_cfg else {}) for obj_name, obj_cfg in (objectives_cfg or {}).items()}
#         self.objectives_cfg =   {
#                                     obj_name: {
#                                         k: obj_cfg[k]
#                                         for k in ("num_classes", "num_species")
#                                         if k in obj_cfg
#                                     }
#                                     for obj_name, obj_cfg in (objectives_cfg or {}).items()
#                                 }

#         # Build encoder
#         self.conv1 = tf.keras.layers.Conv1D(self.conv_channels[0], 3, padding="same", activation="relu")
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.conv2 = tf.keras.layers.Conv1D(self.conv_channels[1], 3, padding="same", activation="relu")
#         self.bn2 = tf.keras.layers.BatchNormalization()

#         # Build heads based on objectives
#         if "polyphony_reg" in self.objectives_cfg:
#             self.segment_pool = tf.keras.layers.GlobalAveragePooling1D()
#             self.segment_dense1 = tf.keras.layers.Dense(128, activation="relu")
#             self.segment_dropout = tf.keras.layers.Dropout(dropout_rate)
#             self.segment_dense2 = tf.keras.layers.Dense(1)

#         if "polyphony_class" in self.objectives_cfg:
#             self.polyphony_num_classes = self.objectives_cfg.get("polyphony_class", {}).get("num_classes", 9)
#             self.segment_pool_class = tf.keras.layers.GlobalAveragePooling1D()
#             self.segment_dense1_class = tf.keras.layers.Dense(128, activation="relu")
#             self.segment_dropout_class = tf.keras.layers.Dropout(dropout_rate)
#             self.polyphony_class_head = tf.keras.layers.Dense(self.polyphony_num_classes)

#         if "event_logits" in self.objectives_cfg:
#             self.event_head = tf.keras.layers.Conv1D(1, kernel_size=1)

#         if "framewise_polyphony_reg" in self.objectives_cfg:
#             self.frame_polyphony_reg_head = tf.keras.layers.Conv1D(1, kernel_size=1)

#         if "framewise_polyphony_class" in self.objectives_cfg:
#             self.frame_polyphony_num_classes = self.objectives_cfg.get("framewise_polyphony_class", {}).get("num_classes", 9)
#             self.frame_polyphony_class_head = tf.keras.layers.Conv1D(self.frame_polyphony_num_classes, kernel_size=1)

#         if "species_polyphony_reg" in self.objectives_cfg:
#             self.num_species_reg = self.objectives_cfg.get("species_polyphony_reg", {}).get("num_species")
#             self.species_polyphony_reg_pool = tf.keras.layers.GlobalAveragePooling1D()
#             self.species_polyphony_reg_dense1 = tf.keras.layers.Dense(256, activation="relu")
#             self.species_polyphony_reg_dropout = tf.keras.layers.Dropout(dropout_rate)
#             self.species_polyphony_reg_dense2 = tf.keras.layers.Dense(128, activation="relu")
#             self.species_polyphony_reg_head = tf.keras.layers.Dense(self.num_species_reg)

#         if "species_polyphony_class" in self.objectives_cfg:
#             self.num_classes_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_classes")
#             self.num_species_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_species")
#             self.species_polyphony_class_pool = tf.keras.layers.GlobalAveragePooling1D()
#             self.species_polyphony_class_dense1 = tf.keras.layers.Dense(256, activation="relu")
#             self.species_polyphony_class_dropout = tf.keras.layers.Dropout(dropout_rate)
#             self.species_polyphony_class_dense2 = tf.keras.layers.Dense(128, activation="relu")
#             self.species_polyphony_class_head = tf.keras.layers.Dense(self.num_species_global * self.num_classes_global)

#         # if "framewise_species_polyphony_reg" in self.objectives_cfg:
#         #     self.num_species_fw = self.objectives_cfg.get("framewise_species_polyphony_reg", {}).get("num_species")
#         #     self.framewise_species_polyphony_reg_head = tf.keras.layers.Conv1D(self.num_species_fw, kernel_size=1)

#         # if "framewise_species_polyphony_class" in self.objectives_cfg:
#         #     self.num_classes_fw = self.objectives_cfg.get("framewise_species_polyphony_class", {}).get("num_classes")
#         #     self.num_species_fw = self.objectives_cfg.get("framewise_species_polyphony_class", {}).get("num_species")
#         #     self.framewise_species_polyphony_class_head = tf.keras.layers.Conv1D(self.num_species_fw * self.num_classes_fw, kernel_size=1)
        

#     def call(self, inputs, training=False):
#         x = tf.reduce_mean(inputs, axis=2)
#         x = self.conv1(x)
#         x = self.bn1(x, training=training)
#         x = self.dropout1(x, training=training)
#         x = self.conv2(x)
#         features = self.bn2(x, training=training)

#         outputs = {}

#         if "polyphony_reg" in self.objectives_cfg:
#             x = self.segment_pool(features)
#             x = self.segment_dense1(x)
#             x = self.segment_dropout(x, training=training)
#             outputs["polyphony_reg"] = self.segment_dense2(x)
            
#         if "polyphony_class" in self.objectives_cfg:
#             x = self.segment_pool_class(features)
#             x = self.segment_dense1_class(x)
#             x = self.segment_dropout_class(x, training=training)
#             outputs["polyphony_class"] = self.polyphony_class_head(x, training=training)

#         if "event_logits" in self.objectives_cfg:
#             outputs["event_logits"] = tf.squeeze(self.event_head(features, training=training), axis=-1)

#         if "framewise_polyphony_reg" in self.objectives_cfg:
#             outputs["framewise_polyphony_reg"] = tf.squeeze(self.frame_polyphony_reg_head(features, training=training), axis=-1)

#         if "framewise_polyphony_class" in self.objectives_cfg:
#             outputs["framewise_polyphony_class"] = self.frame_polyphony_class_head(features, training=training)
        
#         if "species_polyphony_reg" in self.objectives_cfg:
#             x = self.species_polyphony_reg_pool(features)
#             x = self.species_polyphony_reg_dense1(x)
#             x = self.species_polyphony_reg_dropout(x, training=training)
#             x = self.species_polyphony_reg_dense2(x)
#             outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(x)

#         if "species_polyphony_class" in self.objectives_cfg:
#             x = self.species_polyphony_class_pool(features)
#             x = self.species_polyphony_class_dense1(x)
#             x = self.species_polyphony_class_dropout(x, training=training)
#             x = self.species_polyphony_class_dense2(x)
#             x = self.species_polyphony_class_head(x)
#             outputs["species_polyphony_class"] = tf.reshape(
#                 x, (-1, self.num_species_global, self.num_classes_global)
#             )  # shape: (batch, num_species, num_classes)

#         # if "framewise_species_polyphony_reg" in self.objectives_cfg:
#         #     outputs["framewise_species_polyphony_reg"] = self.framewise_species_polyphony_reg_head(
#         #         features, training=training
#         #     )  # shape: (batch, time, num_species)

#         # if "framewise_species_polyphony_class" in self.objectives_cfg:
#         #     x = self.framewise_species_polyphony_class_head(features, training=training)
#         #     batch = tf.shape(x)[0]
#         #     time = tf.shape(x)[1]
#         #     outputs["framewise_species_polyphony_class"] = tf.reshape(
#         #         x, (batch, time, self.num_species_fw, self.num_classes_fw)
#         #     )  # shape: (batch, time, num_species, num_classes)
        
#         return outputs

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "input_dim": self.input_dim,
#             "conv_channels": self.conv_channels,
#             "dropout_rate": self.dropout_rate,
#             "objectives_cfg": self.objectives_cfg,
#         })
#         return config


# @register_keras_serializable(package="model", name="SimpleMLP")
# class SimpleMLP(tf.keras.Model):
#     def __init__(
#         self,
#         input_dim=(None, 4, 1536),
#         hidden_units=[512, 256],
#         dropout_rate=0.3,
#         objectives_cfg=None,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         # Store config
#         self.input_dim = input_dim
#         self.hidden_units = list(hidden_units)
#         self.dropout_rate = dropout_rate
#         self.objectives_cfg =   {
#                                     obj_name: {
#                                         k: obj_cfg[k]
#                                         for k in ("num_classes", "num_species")
#                                         if k in obj_cfg
#                                     }
#                                     for obj_name, obj_cfg in (objectives_cfg or {}).items()
#                                 }

#         # Build encoder (shared feature extraction)
#         self.flatten = layers.Flatten()
        
#         # Hidden layers
#         self.hidden_layers = []
#         self.dropout_layers = []
#         for i, units in enumerate(hidden_units):
#             self.hidden_layers.append(layers.Dense(units, activation='relu'))
#             if i == 0:
#                 self.dropout_layers.append(layers.Dropout(dropout_rate))
#             else:
#                 self.dropout_layers.append(None)
        
#         # Build heads based on objectives
#         if "polyphony_reg" in self.objectives_cfg:
#             self.polyphony_reg_head = layers.Dense(1)

#         if "polyphony_class" in self.objectives_cfg:
#             self.polyphony_num_classes = self.objectives_cfg.get("polyphony_class", {}).get("num_classes", 9) 
#             self.polyphony_class_head = layers.Dense(self.polyphony_num_classes)

#         # if "event_logits" in self.objectives_cfg:
#         #     self.event_head = layers.Dense(1)
        
#         # if "framewise_polyphony_reg" in self.objectives_cfg:
#         #     self.frame_polyphony_reg_head = layers.Dense(1)

#         # if "framewise_polyphony_class" in self.objectives_cfg:
#         #     self.frame_polyphony_class_head = layers.Dense(self.frame_polyphony_num_classes)

#         # Deeper bottleneck than polyphony_reg: joint multi-species prediction
#         # requires capacity to model inter-species correlations
#         if "species_polyphony_reg" in self.objectives_cfg:
#             self.num_species_reg = self.objectives_cfg.get("species_polyphony_reg", {}).get("num_species")
#             self.species_polyphony_reg_dense1 = layers.Dense(256, activation="relu")
#             self.species_polyphony_reg_dropout = layers.Dropout(dropout_rate)
#             self.species_polyphony_reg_dense2 = layers.Dense(128, activation="relu")
#             self.species_polyphony_reg_head = layers.Dense(self.num_species_reg)

#         if "species_polyphony_class" in self.objectives_cfg:
#             self.num_classes_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_classes")
#             self.num_species_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_species")
#             self.species_polyphony_class_dense1 = layers.Dense(256, activation="relu")
#             self.species_polyphony_class_dropout = layers.Dropout(dropout_rate)
#             self.species_polyphony_class_dense2 = layers.Dense(128, activation="relu")
#             self.species_polyphony_class_head = layers.Dense(self.num_species_global * self.num_classes_global)
        
#     # def build(self, input_shape):
#     #     self.input_dim = input_shape
#     #     super().build(input_shape)
    
#     def call(self, inputs, training=False):
#         # Shared encoder
#         x = self.flatten(inputs)
        
#         # Pass through hidden layers
#         for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
#             x = hidden_layer(x)
#             if dropout_layer is not None:
#                 x = dropout_layer(x, training=training)
        
#         # Store shared features
#         features = x
        
#         # Multi-task heads
#         outputs = {}

#         if "polyphony_reg" in self.objectives_cfg:
#             outputs["polyphony_reg"] = self.polyphony_reg_head(features, training=training)

#         if "polyphony_class" in self.objectives_cfg:
#             outputs["polyphony_class"] = self.polyphony_class_head(
#                 features, training=training
#             )
        
#         # if "event_logits" in self.objectives_cfg:
#         #     outputs["event_logits"] = tf.squeeze(self.event_head(features, training=training), axis=-1)
        
#         # if "framewise_polyphony_reg" in self.objectives_cfg:
#         #     outputs["framewise_polyphony_reg"] = tf.squeeze(self.frame_polyphony_reg_head(features, training=training), axis=-1)
        
#         # if "framewise_polyphony_class" in self.objectives_cfg:
#         #     outputs["framewise_polyphony_class"] = self.frame_polyphony_class_head(
#         #         features, training=training
#         #     )

#         if "species_polyphony_reg" in self.objectives_cfg:
#             x = self.species_polyphony_reg_dense1(features)
#             x = self.species_polyphony_reg_dropout(x, training=training)
#             x = self.species_polyphony_reg_dense2(x)
#             outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(x)

#         if "species_polyphony_class" in self.objectives_cfg:
#             x = self.species_polyphony_class_dense1(features)
#             x = self.species_polyphony_class_dropout(x, training=training)
#             x = self.species_polyphony_class_dense2(x)
#             x = self.species_polyphony_class_head(x)
#             outputs["species_polyphony_class"] = tf.reshape(
#                 x, (-1, self.num_species_global, self.num_classes_global)
#             )  # shape: (batch, num_species, num_classes)

#         return outputs
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "input_dim": self.input_dim,
#             "hidden_units": self.hidden_units,
#             "dropout_rate": self.dropout_rate,
#             "objectives_cfg": self.objectives_cfg,
#         })
#         return config



"""
Backbone-agnostic bioacoustics model (TensorFlow / Keras).

Mirrors the torch pattern: SimpleMLP and TemporalCNN are left byte-for-byte
untouched (they're already used directly on precomputed embeddings and must
keep working that way). All backbone-swapping, shape normalization, and
pooled-vs-spatial routing lives in the wrapper below -- neither head knows
or cares which backbone (or none) produced its input.

Division of responsibility:
  - HopliteBackbone: loads any perch_hoplite zoo preset, exposes a pooled
    embedding (batch, dim) via the library's own .embed(), plus a
    canonical 4D spatial embedding (batch, time, spatial, dim). For
    TaxonomyModelTF-backed presets (perch_v2 and family), the spatial
    embedding is TRUE structure recovered by bypassing .embed(), which
    was confirmed via source inspection to unconditionally discard
    'spatial_embedding' from the raw SavedModel output. For presets
    without this path (BirdNET/VGGish), a dummy spatial axis is inserted
    around their native frame sequence instead, same idea as the torch
    wrappers' `spatial_embeddings[:, None, :]`. See
    HopliteBackbone._raw_spatial_embed() and .has_structure().
  - Each head declares a class-level `takes_spatial_embeddings` flag (False
    for SimpleMLP, True for TemporalCNN) -- metadata only, never read inside
    call()/get_config(), so it has no effect on existing standalone use on
    precomputed embeddings.
  - BioacousticsModel: wires backbone -> (pooled or spatial) -> head,
    reading that flag via getattr(head_cls, "takes_spatial_embeddings",
    False) to decide which embedding the head receives.

Precomputed-embedding training is untouched: just keep calling
SimpleMLP(...)/TemporalCNN(...) directly on your precomputed batches, same
as always. This wrapper is only for the new raw-waveform, backbone-attached
path.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

from perch_hoplite.zoo import model_configs


def _to_plain_config(obj):
    """Recursively converts any OmegaConf DictConfig/ListConfig found in
    obj -- including nested inside plain dicts/lists -- to plain dict/list,
    so anything derived from it is JSON-safe for Keras's
    model.save()/get_config(). No-op on values that are already plain
    Python, or if omegaconf isn't installed.

    Needed because SimpleMLP/TemporalCNN happen to already be safe from this
    (their objectives_cfg dict-comprehension in __init__ incidentally
    rebuilds a plain dict), but BioacousticsModel.get_config() stores
    objectives_cfg/head_kwargs directly -- without this, passing Hydra's
    cfg.objectives straight through fails model.save() with "Cannot
    serialize object ... of type DictConfig".

    Recurses into plain dict/list too, not just OmegaConf containers:
    confirmed empirically that `{k: v for k, v in some_dictconfig.items()}`
    produces a plain dict whose VALUES are still un-converted ListConfig/
    DictConfig objects (e.g. hidden_units: [512, 256] stayed a ListConfig
    even after being pulled into a plain dict by train.py's
    build_new_model()) -- checking only the top-level type misses this.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return obj
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _to_plain_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_config(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# 1. Heads -- unchanged from model.py, copied verbatim. Do not edit these;
#    add new head behavior in a new class instead (as with the torch
#    MultiTaskTemporalCNNHead vs. the literal SimpleMLPHead port).
# ---------------------------------------------------------------------------

@register_keras_serializable(package="model", name="SimpleMLP")
class SimpleMLP(tf.keras.Model):
    # Class-level metadata only -- never read inside call()/get_config(), so
    # this has zero effect on standalone use on precomputed embeddings.
    # Consulted externally, e.g. by BioacousticsModel, via getattr(head,
    # "takes_spatial_embeddings", False), mirroring the torch heads' pattern.
    takes_spatial_embeddings = False

    def __init__(
        self,
        input_dim=(None, 4, 1536),
        hidden_units=[512, 256],
        dropout_rate=0.3,
        objectives_cfg=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_units = list(hidden_units)
        self.dropout_rate = dropout_rate
        self.objectives_cfg = {
            obj_name: {
                k: obj_cfg[k]
                for k in ("num_classes", "num_species")
                if k in obj_cfg
            }
            for obj_name, obj_cfg in (objectives_cfg or {}).items()
        }

        self.flatten = layers.Flatten()

        self.hidden_layers = []
        self.dropout_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
            if i == 0:
                self.dropout_layers.append(layers.Dropout(dropout_rate))
            else:
                self.dropout_layers.append(None)

        if "polyphony_reg" in self.objectives_cfg:
            self.polyphony_reg_head = layers.Dense(1)

        if "polyphony_class" in self.objectives_cfg:
            self.polyphony_num_classes = self.objectives_cfg.get("polyphony_class", {}).get("num_classes", 9)
            self.polyphony_class_head = layers.Dense(self.polyphony_num_classes)

        if "species_polyphony_reg" in self.objectives_cfg:
            self.num_species_reg = self.objectives_cfg.get("species_polyphony_reg", {}).get("num_species")
            self.species_polyphony_reg_dense1 = layers.Dense(256, activation="relu")
            self.species_polyphony_reg_dropout = layers.Dropout(dropout_rate)
            self.species_polyphony_reg_dense2 = layers.Dense(128, activation="relu")
            self.species_polyphony_reg_head = layers.Dense(self.num_species_reg)

        if "species_polyphony_class" in self.objectives_cfg:
            self.num_classes_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_classes")
            self.num_species_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_species")
            self.species_polyphony_class_dense1 = layers.Dense(256, activation="relu")
            self.species_polyphony_class_dropout = layers.Dropout(dropout_rate)
            self.species_polyphony_class_dense2 = layers.Dense(128, activation="relu")
            self.species_polyphony_class_head = layers.Dense(self.num_species_global * self.num_classes_global)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)

        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            if dropout_layer is not None:
                x = dropout_layer(x, training=training)

        features = x
        outputs = {}

        if "polyphony_reg" in self.objectives_cfg:
            outputs["polyphony_reg"] = self.polyphony_reg_head(features, training=training)

        if "polyphony_class" in self.objectives_cfg:
            outputs["polyphony_class"] = self.polyphony_class_head(
                features, training=training
            )

        if "species_polyphony_reg" in self.objectives_cfg:
            x = self.species_polyphony_reg_dense1(features)
            x = self.species_polyphony_reg_dropout(x, training=training)
            x = self.species_polyphony_reg_dense2(x)
            outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(x)

        if "species_polyphony_class" in self.objectives_cfg:
            x = self.species_polyphony_class_dense1(features)
            x = self.species_polyphony_class_dropout(x, training=training)
            x = self.species_polyphony_class_dense2(x)
            x = self.species_polyphony_class_head(x)
            outputs["species_polyphony_class"] = tf.reshape(
                x, (-1, self.num_species_global, self.num_classes_global)
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "objectives_cfg": self.objectives_cfg,
        })
        return config


@register_keras_serializable(package="model", name="TemporalCNN")
class TemporalCNN(tf.keras.Model):
    # See SimpleMLP's note above -- class-level metadata only, unread by
    # call()/get_config(); doesn't affect standalone use on precomputed
    # spatial embeddings.
    takes_spatial_embeddings = True

    def __init__(
        self,
        input_dim=(None, 4, 1536),
        conv_channels=(512, 256),
        dropout_rate=0.3,
        objectives_cfg=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.conv_channels = list(conv_channels)
        self.dropout_rate = dropout_rate
        self.objectives_cfg = {
            obj_name: {
                k: obj_cfg[k]
                for k in ("num_classes", "num_species")
                if k in obj_cfg
            }
            for obj_name, obj_cfg in (objectives_cfg or {}).items()
        }

        self.conv1 = tf.keras.layers.Conv1D(self.conv_channels[0], 3, padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(self.conv_channels[1], 3, padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()

        if "polyphony_reg" in self.objectives_cfg:
            self.segment_pool = tf.keras.layers.GlobalAveragePooling1D()
            self.segment_dense1 = tf.keras.layers.Dense(128, activation="relu")
            self.segment_dropout = tf.keras.layers.Dropout(dropout_rate)
            self.segment_dense2 = tf.keras.layers.Dense(1)

        if "polyphony_class" in self.objectives_cfg:
            self.polyphony_num_classes = self.objectives_cfg.get("polyphony_class", {}).get("num_classes", 9)
            self.segment_pool_class = tf.keras.layers.GlobalAveragePooling1D()
            self.segment_dense1_class = tf.keras.layers.Dense(128, activation="relu")
            self.segment_dropout_class = tf.keras.layers.Dropout(dropout_rate)
            self.polyphony_class_head = tf.keras.layers.Dense(self.polyphony_num_classes)

        if "event_logits" in self.objectives_cfg:
            self.event_head = tf.keras.layers.Conv1D(1, kernel_size=1)

        if "framewise_polyphony_reg" in self.objectives_cfg:
            self.frame_polyphony_reg_head = tf.keras.layers.Conv1D(1, kernel_size=1)

        if "framewise_polyphony_class" in self.objectives_cfg:
            self.frame_polyphony_num_classes = self.objectives_cfg.get("framewise_polyphony_class", {}).get("num_classes", 9)
            self.frame_polyphony_class_head = tf.keras.layers.Conv1D(self.frame_polyphony_num_classes, kernel_size=1)

        if "species_polyphony_reg" in self.objectives_cfg:
            self.num_species_reg = self.objectives_cfg.get("species_polyphony_reg", {}).get("num_species")
            self.species_polyphony_reg_pool = tf.keras.layers.GlobalAveragePooling1D()
            self.species_polyphony_reg_dense1 = tf.keras.layers.Dense(256, activation="relu")
            self.species_polyphony_reg_dropout = tf.keras.layers.Dropout(dropout_rate)
            self.species_polyphony_reg_dense2 = tf.keras.layers.Dense(128, activation="relu")
            self.species_polyphony_reg_head = tf.keras.layers.Dense(self.num_species_reg)

        if "species_polyphony_class" in self.objectives_cfg:
            self.num_classes_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_classes")
            self.num_species_global = self.objectives_cfg.get("species_polyphony_class", {}).get("num_species")
            self.species_polyphony_class_pool = tf.keras.layers.GlobalAveragePooling1D()
            self.species_polyphony_class_dense1 = tf.keras.layers.Dense(256, activation="relu")
            self.species_polyphony_class_dropout = tf.keras.layers.Dropout(dropout_rate)
            self.species_polyphony_class_dense2 = tf.keras.layers.Dense(128, activation="relu")
            self.species_polyphony_class_head = tf.keras.layers.Dense(self.num_species_global * self.num_classes_global)

    def call(self, inputs, training=False):
        x = tf.reduce_mean(inputs, axis=2)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        features = self.bn2(x, training=training)

        outputs = {}

        if "polyphony_reg" in self.objectives_cfg:
            x = self.segment_pool(features)
            x = self.segment_dense1(x)
            x = self.segment_dropout(x, training=training)
            outputs["polyphony_reg"] = self.segment_dense2(x)

        if "polyphony_class" in self.objectives_cfg:
            x = self.segment_pool_class(features)
            x = self.segment_dense1_class(x)
            x = self.segment_dropout_class(x, training=training)
            outputs["polyphony_class"] = self.polyphony_class_head(x, training=training)

        if "event_logits" in self.objectives_cfg:
            outputs["event_logits"] = tf.squeeze(self.event_head(features, training=training), axis=-1)

        if "framewise_polyphony_reg" in self.objectives_cfg:
            outputs["framewise_polyphony_reg"] = tf.squeeze(self.frame_polyphony_reg_head(features, training=training), axis=-1)

        if "framewise_polyphony_class" in self.objectives_cfg:
            outputs["framewise_polyphony_class"] = self.frame_polyphony_class_head(features, training=training)

        if "species_polyphony_reg" in self.objectives_cfg:
            x = self.species_polyphony_reg_pool(features)
            x = self.species_polyphony_reg_dense1(x)
            x = self.species_polyphony_reg_dropout(x, training=training)
            x = self.species_polyphony_reg_dense2(x)
            outputs["species_polyphony_reg"] = self.species_polyphony_reg_head(x)

        if "species_polyphony_class" in self.objectives_cfg:
            x = self.species_polyphony_class_pool(features)
            x = self.species_polyphony_class_dense1(x)
            x = self.species_polyphony_class_dropout(x, training=training)
            x = self.species_polyphony_class_dense2(x)
            x = self.species_polyphony_class_head(x)
            outputs["species_polyphony_class"] = tf.reshape(
                x, (-1, self.num_species_global, self.num_classes_global)
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "conv_channels": self.conv_channels,
            "dropout_rate": self.dropout_rate,
            "objectives_cfg": self.objectives_cfg,
        })
        return config


# ---------------------------------------------------------------------------
# 2. Backbone: always exposes pooled_embeddings AND a canonical 4D
#    spatial_embeddings tensor, regardless of what the underlying preset
#    natively returns.
# ---------------------------------------------------------------------------

class HopliteBackbone:
    """Loads a perch_hoplite zoo preset and normalizes its output to a
    fixed contract, mirroring EmbeddingModelOutput on the torch side:

        pooled_embeddings:  (batch, embedding_dim) -- from the library's
                             own .embed(), safe and well-tested.
        spatial_embeddings: (batch, time, spatial, embedding_dim), rank 4.
                             For TaxonomyModelTF-backed presets (perch_v2
                             confirmed; perch_8/surfperch untested -- may
                             hit the older non-batchable path instead),
                             this is TRUE structure recovered by bypassing
                             the library's .embed()/batch_embed(), which
                             unconditionally discards 'spatial_embedding'
                             from the raw SavedModel signature output --
                             confirmed via source inspection, not guessed;
                             see _raw_spatial_embed()'s docstring. For
                             presets without this path (BirdNET/VGGish), a
                             dummy spatial axis is inserted around their
                             native frame sequence, same as the torch
                             wrappers' `[:, None, :]` pattern.
        logits:              raw backbone classification logits, if any --
                             separate from any head's logits.

    Presets that can't produce >1-frame structure by either path get
    spatial_embeddings=None or a degenerate (batch, 1, 1, dim) tensor:
    TemporalCNN genuinely cannot learn anything useful from these. Checked
    via has_structure() (empirical probe, cached) in BioacousticsModel.__init__.
    """

    def __init__(self, preset_name: str, trainable_backbone: bool = False):
        self.preset_name = preset_name
        self._preset_info = model_configs.get_preset_model_config(preset_name)
        self.embedding_dim = self._preset_info.embedding_dim
        self.sample_rate = self._preset_info.model_config.sample_rate
        self.model = self._preset_info.load_model()
        self.set_trainable(trainable_backbone)
        self._has_structure_cache = None

    def set_trainable(self, trainable: bool):
        """Equivalent of the torch wrappers' freeze_encoder()."""
        self.trainable_backbone = trainable
        underlying = getattr(self.model, "model", None)
        if underlying is not None and hasattr(underlying, "trainable"):
            underlying.trainable = trainable
        elif hasattr(self.model, "trainable"):
            self.model.trainable = trainable

    def freeze_encoder(self):
        self.set_trainable(False)

    def has_structure(self) -> bool:
        """Whether this preset can produce a real (>1-frame) spatial
        embedding -- determined empirically via _raw_embed(). Cached.

        NOTE: confirmed via isolated testing (bypassing HopliteBackbone
        entirely) that perch_hoplite's own top-level .embed() has a bug --
        it raises "cannot reshape array of size N into shape (1, 1, ...)"
        for ANY multi-hop input, even at batch=1. Not a batching issue.
        This is why _raw_embed() bypasses .embed()/batch_embed() completely
        for presets where possible, rather than only bypassing it for the
        spatial tensor -- calling the library's .embed() at all is unsafe
        once more than one hop is involved.
        """
        if self._has_structure_cache is not None:
            return self._has_structure_cache

        cfg = self._preset_info.model_config
        window_s = cfg.get("window_size_s", 5.0)
        hop_s = cfg.get("hop_size_s", window_s) or window_s
        probe_duration = window_s + 2 * hop_s
        probe_waveform = tf.zeros((1, int(self.sample_rate * probe_duration)), dtype=tf.float32)

        raw = self._raw_embed(probe_waveform)
        spatial = raw["spatial_embeddings"] if raw is not None else None
        structured = spatial is not None and spatial.shape[1] > 1
        self._has_structure_cache = structured
        return structured

    def _raw_embed(self, waveform: tf.Tensor):
        """Full bypass of TaxonomyModelTF.batch_embed()/.embed() -- computes
        pooled_embeddings, spatial_embeddings, AND logits from ONE raw
        SavedModel signature call, using the wrapper's own frame_audio()/
        normalize_audio() so behavior matches the library exactly.

        This exists for two reasons, confirmed via source inspection and
        isolated testing (not guessed):
          1. batch_embed() unconditionally discards 'spatial_embedding'
             from the raw output before it ever reaches InferenceOutputs.
          2. The library's own top-level .embed() has a separate bug: it
             raises a reshape error for ANY input spanning more than one
             hop, even at batch=1 -- confirmed independent of our wrapper.
             So once more than one hop might be involved, calling the
             library's .embed() at all is unsafe, not just insufficient.

        Per-window spatial grid is e.g. (16, 4, 1536) for perch_v2 (NOT the
        (5, 3, 1536) some docs describe -- stale for this checkpoint).
        Hop-windows (n_hops) and within-window time (within_time) are both
        sequential positions in time, so they're merged into one axis:
            (batch, n_hops, within_time, within_freq, dim)
                -> (batch, n_hops * within_time, within_freq, dim)

        Logits are mean-pooled over hops (one prediction per clip, not
        per-hop) -- a deliberate simplification; per-hop logits would need
        an extra axis this wrapper doesn't currently expose.

        Returns None if this preset/path doesn't support the bypass at all
        (non-batchable models -- older infer_tf path -- or a signature
        missing 'embedding' entirely); callers fall back to self.model.embed(),
        which is safe as long as they never feed it more than one hop.
        """
        wrapper = self.model
        required = ("frame_audio", "normalize_audio", "window_size_s", "hop_size_s", "target_peak")
        if not all(hasattr(wrapper, a) for a in required):
            return None
        if not getattr(wrapper, "batchable", False):
            return None  # older infer_tf path -- unsupported by this fast path
        underlying = getattr(wrapper, "model", None)
        if underlying is None or not hasattr(underlying, "signatures"):
            return None

        audio_np = waveform.numpy() if hasattr(waveform, "numpy") else np.asarray(waveform)
        framed_audio = wrapper.frame_audio(audio_np, wrapper.window_size_s, wrapper.hop_size_s)
        framed_audio = wrapper.normalize_audio(framed_audio, wrapper.target_peak)
        rebatched = framed_audio.reshape([-1, framed_audio.shape[-1]])
        batch, n_hops = framed_audio.shape[0], framed_audio.shape[1]

        try:
            raw_outputs = underlying.signatures['serving_default'](inputs=rebatched)
        except Exception:
            return None
        if 'embedding' not in raw_outputs:
            return None

        pooled = raw_outputs['embedding'].numpy().reshape(batch, n_hops, -1)
        pooled = pooled.mean(axis=1)  # mean over hops -> (batch, dim)

        spatial = None
        if 'spatial_embedding' in raw_outputs:
            sp = raw_outputs['spatial_embedding'].numpy()
            within_time, within_freq, dim = sp.shape[1:]
            sp = sp.reshape(batch, n_hops, within_time, within_freq, dim)
            spatial = tf.constant(sp.reshape(batch, n_hops * within_time, within_freq, dim))

        skip_keys = {'embedding', 'spatial_embedding', 'frontend', 'spectrogram'}
        logits = {}
        for key, val in raw_outputs.items():
            if key in skip_keys:
                continue
            v = val.numpy().reshape(batch, n_hops, -1).mean(axis=1)
            logits[key] = v

        return {
            "pooled_embeddings": tf.constant(pooled),
            "spatial_embeddings": spatial,
            "logits": logits,
        }

    def get_head_input_size(self, pooled: bool):
        return self.embedding_dim if pooled else (None, None, None, self.embedding_dim)

    def embed(self, waveform: tf.Tensor) -> dict:
        """Returns {'pooled_embeddings', 'spatial_embeddings', 'logits'}.

        Prefers the raw-signature bypass (_raw_embed) when available --
        both to recover true spatial structure and to avoid a confirmed
        bug in the library's own .embed() for multi-hop input. Falls back
        to the library's .embed() only for presets without the bypass
        (BirdNET/VGGish/YAMNet) -- callers should avoid feeding those more
        than one hop's worth of audio until that path is verified safe too.
        """
        raw = self._raw_embed(waveform)
        if raw is not None:
            return raw

        outputs = self.model.embed(waveform)
        pooled = outputs.embeddings
        logits = getattr(outputs, "logits", None)

        rank = len(pooled.shape)
        spatial = None
        if rank == 4:
            spatial = None  # library's own (batch, n_hops, 1, dim) has no real structure
            pooled = tf.reduce_mean(pooled, axis=[1, 2])
        elif rank == 3:
            spatial = pooled[:, :, tf.newaxis, :]
            pooled = tf.reduce_mean(pooled, axis=1)
        # rank == 2: already pooled, spatial stays None

        return {"pooled_embeddings": pooled, "spatial_embeddings": spatial, "logits": logits}


# ---------------------------------------------------------------------------
# 3. Backbone + head, wired via each head's `takes_spatial_embeddings` flag
# ---------------------------------------------------------------------------

@register_keras_serializable(package="model", name="BioacousticsModel")
class BioacousticsModel(tf.keras.Model):
    """backbone_name (perch_hoplite preset) + head_cls (SimpleMLP/TemporalCNN,
    unmodified) -> full end-to-end waveform model.

    For training heads on precomputed embeddings, skip this class entirely
    and keep calling SimpleMLP(...)/TemporalCNN(...) directly, as before.
    """

    # Serialization can't pickle a class object (head_cls) directly into a
    # JSON-safe Keras config -- get_config()/from_config() store/resolve the
    # head by name through this registry instead. Add new heads here if you
    # add new head classes intended for use with BioacousticsModel.
    _HEAD_REGISTRY = {"SimpleMLP": SimpleMLP, "TemporalCNN": TemporalCNN}

    def __init__(self, backbone_name: str, head_cls=SimpleMLP,
                 objectives_cfg=None, head_kwargs=None,
                 trainable_backbone: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.backbone = HopliteBackbone(backbone_name, trainable_backbone=trainable_backbone)

        # Convert once, up front, so everything downstream (head construction
        # AND what gets stashed for get_config()) works with plain Python,
        # regardless of whether the caller passed Hydra's OmegaConf objects
        # or already-plain dicts/lists.
        objectives_cfg = _to_plain_config(objectives_cfg)
        head_kwargs = _to_plain_config(head_kwargs)

        if not (isinstance(head_cls, type) and issubclass(head_cls, tf.keras.Model)):
            raise TypeError(
                f"head_cls={head_cls!r} isn't a tf.keras.Model subclass. This "
                f"usually means cfg.head._target_ points at a torch class "
                f"(e.g. 'utils.torch_models.SimpleMLPHead') instead of the TF "
                f"one -- use 'model.SimpleMLP' or 'model.TemporalCNN'."
            )

        wants_spatial = getattr(head_cls, "takes_spatial_embeddings", False)
        if wants_spatial and not self.backbone.has_structure():
            raise ValueError(
                f"'{backbone_name}' has no time/spatial structure (the library "
                f"pools it internally), so '{head_cls.__name__}' can't be used "
                f"with it. Use SimpleMLP instead, or pick a backbone that "
                f"preserves structure (e.g. 'perch_v2', 'birdnet_V2.4', 'vggish')."
            )
        self._wants_spatial = wants_spatial

        resolved_head_kwargs = dict(head_kwargs or {})
        # objectives_cfg is always supplied by BioacousticsModel itself (the
        # parameter above), never by cfg.head -- if it's also in head_kwargs
        # (a common copy-paste mistake from a torch-shaped head config, whose
        # head class embeds objectives_cfg as a constructor field), that's a
        # config error worth surfacing clearly rather than crashing on a
        # duplicate-keyword TypeError two frames deeper.
        if "objectives_cfg" in resolved_head_kwargs:
            raise ValueError(
                "head_kwargs already contains 'objectives_cfg' -- remove it "
                "from cfg.head; BioacousticsModel supplies it separately from "
                "cfg.objectives. (This usually means cfg.head was copied from "
                "a torch-style head config, whose head class embeds "
                "objectives_cfg as a constructor field -- the TF heads don't.)"
            )
        resolved_head_kwargs.setdefault("input_dim", self.backbone.get_head_input_size(pooled=not wants_spatial))
        self.head = head_cls(objectives_cfg=objectives_cfg, **resolved_head_kwargs)

        # Stashed verbatim for get_config() -- resolved_head_kwargs already
        # has input_dim filled in, so from_config() reconstructs the exact
        # same head shape without needing to reload the backbone twice to
        # recompute it.
        if head_cls.__name__ not in self._HEAD_REGISTRY:
            raise ValueError(
                f"'{head_cls.__name__}' isn't in BioacousticsModel._HEAD_REGISTRY, "
                f"so this model can be trained but won't survive "
                f"tf.keras.models.load_model() round-tripping. Add it to the "
                f"registry if you need that."
            )
        self._head_cls_name = head_cls.__name__
        self._objectives_cfg = objectives_cfg
        self._resolved_head_kwargs = resolved_head_kwargs

    @property
    def sample_rate(self):
        return self.backbone.sample_rate

    def call(self, waveform, training=False):
        embeds = self.backbone.embed(waveform)
        head_input = embeds["spatial_embeddings"] if self._wants_spatial else embeds["pooled_embeddings"]
        return self.head(head_input, training=training)

    def set_backbone_trainable(self, trainable: bool):
        """NOTE: call model.compile(...) again after this -- Keras fixes the
        trainable-variable list at compile time, unlike torch's requires_grad,
        which the optimizer re-reads from param groups on the fly."""
        self.backbone.set_trainable(trainable)

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone_name": self.backbone_name,
            "head_cls_name": self._head_cls_name,
            "objectives_cfg": self._objectives_cfg,
            "head_kwargs": self._resolved_head_kwargs,
            # Current state, not the value passed at construction -- so
            # saving a model after set_backbone_trainable(True) (e.g. after
            # the freeze_epochs unfreeze point) round-trips correctly on load,
            # rather than reverting to whatever it was frozen as initially.
            "trainable_backbone": self.backbone.trainable_backbone,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        head_cls_name = config.pop("head_cls_name")
        if head_cls_name not in cls._HEAD_REGISTRY:
            raise ValueError(
                f"Unknown head class '{head_cls_name}' in saved config -- "
                f"known heads: {list(cls._HEAD_REGISTRY)}. If you renamed or "
                f"added a head class since this model was saved, update "
                f"BioacousticsModel._HEAD_REGISTRY to match."
            )
        head_cls = cls._HEAD_REGISTRY[head_cls_name]
        # Base tf.keras.Model config keys (name, trainable, dtype, ...) pass
        # straight through via **config; only head_cls needs resolving above.
        return cls(head_cls=head_cls, **config)


def build_new_model(precomputed_embeddings, model_cfg, backbone_cfg, head_cfg,
                     objectives_cfg, freeze_encoder):
    """Returns an uncompiled, freshly-instantiated model.

    precomputed_embeddings=True: unchanged behavior -- cfg.model._target_
    (e.g. model.SimpleMLP) is instantiated directly as the whole model,
    exactly as in existing configs/experiments that train on precomputed
    embeddings.

    precomputed_embeddings=False: builds a BioacousticsModel wrapping a
    perch_hoplite backbone (cfg.backbone.name) and a head (cfg.head._target_,
    e.g. model.TemporalCNN). The head class is resolved via get_class()
    rather than instantiate(), since BioacousticsModel needs the class
    itself (it builds the head internally once it knows the backbone's
    output shape) -- the rest of cfg.head's fields are passed through as
    head_kwargs.

    Callers derive precomputed_embeddings from cfg.train.backend (== 'perch'
    means False), the same way evaluate_on_test_split.py already checks
    backend == 'torch' -- this function itself doesn't know about 'backend'
    at all, just which of the two branches to build, mirroring
    torch_train.py's precomputed_embeddings/freeze_encoder naming.

    Shared by train.py and evaluate_on_test_split.py -- previously
    duplicated in both; keep it here as the single source of truth.
    """
    from hydra.utils import instantiate, get_class

    if precomputed_embeddings:
        return instantiate(model_cfg)

    head_target = head_cfg._target_
    head_cls = get_class(head_target)
    head_kwargs = {k: v for k, v in head_cfg.items() if k != "_target_"}
    return BioacousticsModel(
        backbone_name=backbone_cfg.name,
        head_cls=head_cls,
        objectives_cfg=objectives_cfg,
        head_kwargs=head_kwargs,
        trainable_backbone=not freeze_encoder,
    )


# ---------------------------------------------------------------------------
# 4. Usage
# ---------------------------------------------------------------------------

"""
objectives_cfg = {
    "polyphony_class": {"num_classes": 9},
    "species_polyphony_class": {"num_classes": 5, "num_species": 20},
}

# Existing workflow, unchanged: heads on precomputed embeddings
head = SimpleMLP(objectives_cfg=objectives_cfg)
head(precomputed_pooled_batch)  # (batch, 1536)

head = TemporalCNN(objectives_cfg=objectives_cfg)
head(precomputed_spatial_batch)  # (batch, 5, 3, 1536)

# New: end-to-end waveform model, any backbone
model = BioacousticsModel('perch_v2', head_cls=TemporalCNN, objectives_cfg=objectives_cfg)
model = BioacousticsModel('birdnet_V2.4', head_cls=TemporalCNN, objectives_cfg=objectives_cfg)  # dummy spatial axis inserted
model = BioacousticsModel('vggish', head_cls=SimpleMLP, objectives_cfg=objectives_cfg)

# IMPORTANT: has_structure() only tells you whether a preset CAN produce
# >1-frame output given enough audio -- it doesn't mean every call will.
# If you build with TemporalCNN but then call the model on exactly one
# window_size_s-length clip, you'll still get a degenerate (batch, 1, 1, dim)
# spatial tensor and TemporalCNN will silently run on a length-1 sequence
# (valid Keras, not remotely what you want). Feed audio spanning multiple
# hop_size_s windows if you actually need temporal structure.

# Which presets actually fail has_structure() is now unconfirmed -- the
# smoke test's static exclusion list (perch_8, surfperch) turned out to be
# wrong (see model.py history); re-run smoke_test_backbones.py with the
# updated has_structure() probe to find real examples, if any exist among
# these presets.

# Staged fine-tuning -- recompile after toggling trainable
model.set_backbone_trainable(False)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=..., metrics=...)
model.fit(train_ds, epochs=10)

model.set_backbone_trainable(True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=..., metrics=...)  # required after trainable toggle
model.fit(train_ds, epochs=5)
"""