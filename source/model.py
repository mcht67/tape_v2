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
            "polyphony_degree": count,
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
    
@register_keras_serializable(package="model", name="TemporalCNN")
class TemporalCNN(tf.keras.Model):
    def __init__(
        self,
        input_dim=(None, 4, 1536),
        conv_channels=(512, 256),
        dropout_rate=0.3,
        objectives_cfg=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Store config
        self.input_dim = input_dim
        self.conv_channels = list(conv_channels)
        self.dropout_rate = dropout_rate
        # self.objectives_cfg = {obj_name: ({"num_classes": obj_cfg["num_classes"]} if "num_classes" in obj_cfg else {}) for obj_name, obj_cfg in (objectives_cfg or {}).items()}
        self.objectives_cfg =   {
                                    obj_name: {
                                        k: obj_cfg[k]
                                        for k in ("num_classes", "num_species")
                                        if k in obj_cfg
                                    }
                                    for obj_name, obj_cfg in (objectives_cfg or {}).items()
                                }

        # Build encoder
        self.conv1 = tf.keras.layers.Conv1D(self.conv_channels[0], 3, padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(self.conv_channels[1], 3, padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Build heads based on objectives
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

        # if "framewise_species_polyphony_reg" in self.objectives_cfg:
        #     self.num_species_fw = self.objectives_cfg.get("framewise_species_polyphony_reg", {}).get("num_species")
        #     self.framewise_species_polyphony_reg_head = tf.keras.layers.Conv1D(self.num_species_fw, kernel_size=1)

        # if "framewise_species_polyphony_class" in self.objectives_cfg:
        #     self.num_classes_fw = self.objectives_cfg.get("framewise_species_polyphony_class", {}).get("num_classes")
        #     self.num_species_fw = self.objectives_cfg.get("framewise_species_polyphony_class", {}).get("num_species")
        #     self.framewise_species_polyphony_class_head = tf.keras.layers.Conv1D(self.num_species_fw * self.num_classes_fw, kernel_size=1)
        

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
            )  # shape: (batch, num_species, num_classes)

        # if "framewise_species_polyphony_reg" in self.objectives_cfg:
        #     outputs["framewise_species_polyphony_reg"] = self.framewise_species_polyphony_reg_head(
        #         features, training=training
        #     )  # shape: (batch, time, num_species)

        # if "framewise_species_polyphony_class" in self.objectives_cfg:
        #     x = self.framewise_species_polyphony_class_head(features, training=training)
        #     batch = tf.shape(x)[0]
        #     time = tf.shape(x)[1]
        #     outputs["framewise_species_polyphony_class"] = tf.reshape(
        #         x, (batch, time, self.num_species_fw, self.num_classes_fw)
        #     )  # shape: (batch, time, num_species, num_classes)
        
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


@register_keras_serializable(package="model", name="SimpleMLP")
class SimpleMLP(tf.keras.Model):
    def __init__(
        self,
        input_dim=(None, 4, 1536),
        hidden_units=[512, 256],
        dropout_rate=0.3,
        objectives_cfg=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Store config
        self.input_dim = input_dim
        self.hidden_units = list(hidden_units)
        self.dropout_rate = dropout_rate
        self.objectives_cfg =   {
                                    obj_name: {
                                        k: obj_cfg[k]
                                        for k in ("num_classes", "num_species")
                                        if k in obj_cfg
                                    }
                                    for obj_name, obj_cfg in (objectives_cfg or {}).items()
                                }

        # Build encoder (shared feature extraction)
        self.flatten = layers.Flatten()
        
        # Hidden layers
        self.hidden_layers = []
        self.dropout_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
            if i == 0:
                self.dropout_layers.append(layers.Dropout(dropout_rate))
            else:
                self.dropout_layers.append(None)
        
        # Build heads based on objectives
        if "polyphony_reg" in self.objectives_cfg:
            self.polyphony_reg_head = layers.Dense(1)

        if "polyphony_class" in self.objectives_cfg:
            self.polyphony_num_classes = self.objectives_cfg.get("polyphony_class", {}).get("num_classes", 9) 
            self.polyphony_class_head = layers.Dense(self.polyphony_num_classes)

        if "event_logits" in self.objectives_cfg:
            self.event_head = layers.Dense(1)
        
        if "framewise_polyphony_reg" in self.objectives_cfg:
            self.frame_polyphony_reg_head = layers.Dense(1)

        if "framewise_polyphony_class" in self.objectives_cfg:
            self.frame_polyphony_class_head = layers.Dense(self.frame_polyphony_num_classes)

        # Deeper bottleneck than polyphony_reg: joint multi-species prediction
        # requires capacity to model inter-species correlations
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
        
    # def build(self, input_shape):
    #     self.input_dim = input_shape
    #     super().build(input_shape)
    
    def call(self, inputs, training=False):
        # Shared encoder
        x = self.flatten(inputs)
        
        # Pass through hidden layers
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            if dropout_layer is not None:
                x = dropout_layer(x, training=training)
        
        # Store shared features
        features = x
        
        # Multi-task heads
        outputs = {}

        if "polyphony_reg" in self.objectives_cfg:
            outputs["polyphony_reg"] = self.polyphony_reg_head(features, training=training)

        if "polyphony_class" in self.objectives_cfg:
            outputs["polyphony_class"] = self.polyphony_class_head(
                features, training=training
            )
        
        if "event_logits" in self.objectives_cfg:
            outputs["event_logits"] = tf.squeeze(self.event_head(features, training=training), axis=-1)
        
        if "framewise_polyphony_reg" in self.objectives_cfg:
            outputs["framewise_polyphony_reg"] = tf.squeeze(self.frame_polyphony_reg_head(features, training=training), axis=-1)
        
        if "framewise_polyphony_class" in self.objectives_cfg:
            outputs["framewise_polyphony_class"] = self.frame_polyphony_class_head(
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
            )  # shape: (batch, num_species, num_classes)

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






