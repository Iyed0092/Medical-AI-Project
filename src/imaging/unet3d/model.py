import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPool3D(2)(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv3DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def get_unet3d(input_shape=(64,64,64,1), base_filters=16):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, base_filters)
    c2, p2 = encoder_block(p1, base_filters*2)
    c3, p3 = encoder_block(p2, base_filters*4)
    c4, p4 = encoder_block(p3, base_filters*8)
    b = conv_block(p4, base_filters*16)
    d4 = decoder_block(b, c4, base_filters*8)
    d3 = decoder_block(d4, c3, base_filters*4)
    d2 = decoder_block(d3, c2, base_filters*2)
    d1 = decoder_block(d2, c1, base_filters)
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(d1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    m = get_unet3d()
    m.summary()
