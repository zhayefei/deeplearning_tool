#! encoding=utf-8
import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.engine.topology import Input
from sklearn.metrics import accuracy_score

from layers import GatedLayer

with np.load("mnist.npz") as data:
    for key, value in data.items():
        train_images, train_labels = data['x_train'], data['y_train']
        test_images, test_labels = data['x_test'], data['y_test']

train_labels = train_labels[:1000]
test_labels = test_labels[:100]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:100].reshape(-1, 28 * 28) / 255.0

custom_objects={
    'GatedLayer': GatedLayer,
}

# 定义一个简单的序列模型
def create_model_tf_keras():
    """
    using tf.keras interface
    """
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def create_model():
    """
    using pure keras interface
    """
    from keras.models import Model
    from keras.layers.core import Dropout, Dense
    input = Input(shape=(784,))
    tens = Dense(512, activation=u'relu')(input)
    tens = Dropout(0.2)(tens)
    tens = GatedLayer(output_dim=512)(tens)
    tens = Dense(10)(tens)
    model = Model(inputs=[input], outputs=tens)
    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def train_and_save(save_fmt):
    # 使用新的回调训练模型
    model = create_model()
    model.fit(train_images,
          train_labels,
          epochs=20,
          validation_data=(test_images,test_labels))
    #评估模型
    loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Trained model, accuracy: {:5.2f}%".format(100*acc))
    
    if save_fmt == 'h5':
        model.save("model.h5")
        print("save as model.h5")
    elif save_fmt == 'savedmodel':
        model.save('saved_model') 
        print("save as savedmodel fmt")
        #tf.saved_model.save(model, 'model/saved_model/')
    return model


def predict_using_h5():
    model = keras.models.load_model('model.h5', custom_objects=custom_objects)
    #model.load_weights(checkpoint_path)
    #model.load_weights('training/my_checkpoint')
    #model = tf.keras.models.load_model('saved_model/my_model')
    
    model.summary()
    loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Eval model, accuracy: {:5.2f}%".format(100*acc))

def predict_using_savedmodel():
    model = tf.saved_model.load('saved_model')
    infer = model.signatures["serving_default"]

    input_x = tf.convert_to_tensor(test_images, dtype=float)
    result = infer(input_x)['dense_1']
    # 评估模型
    result = tf.math.argmax(result, 1).numpy()
    accu = accuracy_score(result, test_labels)
    print("Eval model, accuracy:", accu)

def predict(save_fmt):
    if save_fmt == 'h5':
        predict_using_h5()
    else:
        predict_using_savedmodel()

if __name__ == '__main__':
    mode = sys.argv[1]
    save_fmt = sys.argv[2]
    if mode == 'train':
        print("training...")
        train_and_save(save_fmt)
    else:
        predict(save_fmt)
