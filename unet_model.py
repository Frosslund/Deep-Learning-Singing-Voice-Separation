import tensorflow as tf


def u_net(img_width, img_height, img_channels, metrics, batch_norm, filters, optimizer='adam', loss_fn='binary'):

    def contract_layer(inputs, filters, dropout, drop, batch_norm, act='relu', kernel=(5, 5), pool_kernel=(2, 2)):
        use_bias = False if batch_norm else True
        conv = tf.keras.layers.Conv2D(filters, kernel, use_bias=use_bias, kernel_initializer='he_normal', padding='same')(inputs)
        conv = tf.keras.layers.BatchNormalization()(conv) if batch_norm else conv
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        conv = tf.keras.layers.Dropout(drop)(conv) if dropout else conv
        conv = tf.keras.layers.Conv2D(filters, kernel, use_bias=use_bias, kernel_initializer='he_normal', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv) if batch_norm else conv
        conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
        pool = tf.keras.layers.MaxPooling2D(pool_size=pool_kernel)(conv)
        return conv, pool

    def expanse_layer(inputs, filters, conc, dropout, drop, batch_norm, act='relu', trans_kernel=(5, 5), strides=(2, 2), kernel=(5, 5), axis=-1):
        use_bias = False if batch_norm else True
        up = tf.keras.layers.Conv2DTranspose(filters, trans_kernel, strides=strides, padding='same')(inputs)
        up = tf.keras.layers.concatenate([up, conc], axis=axis)
        conv = tf.keras.layers.Conv2D(filters, kernel, use_bias=use_bias, kernel_initializer='he_normal', padding='same')(up)
        conv = tf.keras.layers.BatchNormalization()(conv) if batch_norm else conv
        conv = tf.keras.layers.Activation(act)(conv)
        conv = tf.keras.layers.Dropout(drop)(conv) if dropout else conv
        conv = tf.keras.layers.Conv2D(filters, kernel, use_bias=use_bias, kernel_initializer='he_normal', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv) if batch_norm else conv
        conv = tf.keras.layers.Activation(act)(conv)
        return conv

    # initialize inputs
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

    # contractive path
    c1, p1 = contract_layer(inputs=inputs, filters=filters,
                            dropout=True, drop=0.5, batch_norm=batch_norm)
    c2, p2 = contract_layer(inputs=p1, filters=2*filters,
                            dropout=True, drop=0.5, batch_norm=batch_norm)
    c3, p3 = contract_layer(inputs=p2, filters=4*filters,
                            dropout=True, drop=0.5, batch_norm=batch_norm)
    c4, p4 = contract_layer(inputs=p3, filters=8*filters,
                            dropout=False, drop=0.5, batch_norm=batch_norm)
    c5, _ = contract_layer(inputs=p4, filters=16*filters,
                           dropout=False, drop=0.5, batch_norm=batch_norm)

    # expansive path
    c6 = expanse_layer(inputs=c5, filters=8*filters, conc=c4,
                       dropout=False, drop=0.5, batch_norm=batch_norm)
    c7 = expanse_layer(inputs=c6, filters=4*filters, conc=c3,
                       dropout=False, drop=0.5, batch_norm=batch_norm)
    c8 = expanse_layer(inputs=c7, filters=2*filters, conc=c2,
                       dropout=False, drop=0.5, batch_norm=batch_norm)
    c9 = expanse_layer(inputs=c8, filters=filters, conc=c1,
                       dropout=False, drop=0.5, batch_norm=batch_norm, axis=3)

    # initialize outputs
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # create model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    loss_fns = {'binary': 'binary_crossentropy'}
    loss = loss_fns.get(loss_fn)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
