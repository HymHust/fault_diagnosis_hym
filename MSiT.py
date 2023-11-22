import tensorflow as tf
#from keras.utils import plot_model
from tensorflow.keras import Model, layers, initializers

num_classes = 4 # 类别数
signal_size = [5120,4] # 输入的信号形状
patch_size = [512,4]  # Patch的形状
num_patches = (signal_size[0] // patch_size[0]) *(signal_size[1] // patch_size[1])
projection_dim = 64
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Transformer block的个数
transformer_layers = 10
mlp_head_units = [300, 100]  # 输出部分的MLP全连接层的大小


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, signal):
        batch_size = tf.shape(signal)[0]
        signal = tf.expand_dims(signal, axis=-1)
        patches = tf.image.extract_patches(
            images=signal,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):
        config = {"patch_size": self.patch_size}
        base_config = super(Patches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConcatClassTokenAddPosEmbed(layers.Layer):
    def __init__(self, embed_dim=40, num_patches=40, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def build(self, input_shape):
        self.cls_token = self.add_weight(name="cls",
                                         shape=[1,1, self.embed_dim],
                                         initializer=initializers.Zeros(),
                                         trainable=True,
                                         dtype=tf.float32)
        self.pos_embed = self.add_weight(name="pos_embed",
                                         shape=[1,self.num_patches + 1, self.embed_dim],
                                         initializer=initializers.RandomNormal(stddev=0.02),
                                         trainable=True,
                                         dtype=tf.float32)

    def call(self, inputs, **kwargs):
        batch_size= tf.shape(inputs)[0]
        cls_token = tf.broadcast_to(self.cls_token, shape=[batch_size, 1, self.embed_dim])
        x = tf.concat([cls_token, inputs], axis=1)
        x = x + self.pos_embed

        return x
    def get_config(self):
        config = {"embed_dim": self.embed_dim,"num_patches":self.num_patches}
        base_config = super(ConcatClassTokenAddPosEmbed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = {"num_patches": self.num_patches,"projection":self.projection,'position_embedding':self.position_embedding}
        base_config = super(PatchEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def MSiT(input_shape):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    add_pos_embed=ConcatClassTokenAddPosEmbed(embed_dim=projection_dim,num_patches=num_patches)(encoded_patches)
    # 创建多个Transformer encoding 块
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(add_pos_embed)

        # 创建多头自注意力机制 multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        x2 = layers.Add()([attention_output, add_pos_embed])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.01)
        add_pos_embed = layers.Add()([x3, x2])

    #representation = layers.LayerNormalization(epsilon=1e-6)
    representation = layers.Flatten()(add_pos_embed)
    #representation = layers.Dropout(0.1)(representation)
    # 增加MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.01)
    # 输出分类.
    logits = layers.Dense(units=num_classes, activation='sigmoid')(features)
    # 构建
    model = Model(inputs=inputs, outputs=logits)
    model.summary()
    #plot_model(model, to_file='model_cnn.png', show_shapes=True, show_layer_names='False', rankdir='TB')
    return model


if __name__ == "__main__":
    Model = MSiT([5120,4])


