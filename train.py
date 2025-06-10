import tensorflow as tf
import numpy as np
import json
import os

MODEL_NAME = 'fashion_mnist'
MODEL_DIR = f'model/{MODEL_NAME}'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_H5 = f'{MODEL_DIR}.h5'
MODEL_JSON = f'{MODEL_DIR}.json'
MODEL_NPZ = f'{MODEL_DIR}.npz'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
    tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(64, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(10, activation='softmax', name='output')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test accuracy = {acc:.4f}")

model.save(MODEL_H5)

model_arch = []
for layer in model.layers:
    layer_type = layer.__class__.__name__  
    config = {}

    if hasattr(layer, 'activation') and layer.activation:
        act_name = layer.activation.__name__
        if act_name != 'linear':  
            config['activation'] = act_name

    weight_names = [w.name for w in layer.weights]

    model_arch.append({
        'name': layer.name,
        'type': layer_type,
        'config': config,
        'weights': weight_names
    })

with open(MODEL_JSON, 'w') as f:
    json.dump(model_arch, f, indent=2)

weights = {}
for layer in model.layers:
    for w_tensor, value in zip(layer.weights, layer.get_weights()):
        weights[w_tensor.name] = value
np.savez(MODEL_NPZ, **weights)
