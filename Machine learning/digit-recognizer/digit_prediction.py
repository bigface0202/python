from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D#畳み込みの処理やプーリングなどの処理を行う
from keras.layers import Activation, Dropout, Flatten, Dense #Flatten：データを1次元にする,Dense:全結合層を連結する
from keras.utils import np_utils
import keras
import tensorflow as tf
import numpy as np
import pandas as pd

num_classes = 10

def main():
    test_data = pd.read_csv("./test.csv")
    
    X_test = test_data.values.astype('float32')
    X_test_imgs = X_test.reshape((-1, 28, 28, 1)) / 255.
    
    model = myModel()
    model.load_weights('digit_recog_cnn2.h5')
    results = model_predict(model, X_test_imgs)
    results = np.argmax(results, axis = 1)
    
    result_dict = {
        'ImageId':np.arange(1, len(results) + 1),
        'Label': results
    }
    
    df = pd.DataFrame(result_dict)
    df.to_csv("./result_from_prediction.csv", index_label = False, index = False)
    print("Prediction is done")

def myModel():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(3, 3))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    
    model.add(Activation('softmax'))
    
    return model


def model_predict(model, X):
    test_y = model.predict(X)
    return test_y

if __name__ == "__main__":
    main()