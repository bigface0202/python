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
    X_train_imgs, X_test_imgs, y_train, _ = np.load("./digit_aug_jupyter.npy", allow_pickle=True)
    
    #データオーグメンッテッドの時にデータの形が戻ってしまう(42000, 784)
    #標準化しておく
    #↓いらないかも？
    X_train_imgs = X_train_imgs / 255.
    X_train_imgs = X_train_imgs.astype("float32")
    
    #カテゴリカルな値にしておく
    y_train = np_utils.to_categorical(y_train, num_classes)
    
    # X_test_imgs = X_test_imgs.values.astype('float32')
    X_test_imgs = X_test_imgs / 255.
    
    
    print(X_train_imgs.ndim)
    
    model = model_train(X_train_imgs, y_train)
    result_org = model_predict(model, X_test_imgs)
    result_submit = np.argmax(result_org, axis = 1)
    
    result_dict = {
        'ImageId':np.arange(1, len(result_org) + 1),
        'Label': result_submit
    }
    
    df = pd.DataFrame(result_dict)
    df.to_csv("./result.csv", index_label = False, index = False)
    print("Prediction is done")
    
def model_train(X, y):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = X.shape[1:]))
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
    
    opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = opt,
                  metrics = ['accuracy'])
    #エポック数上げると性能が良くなるらしい（このNNの場合）
    model.fit(X, y, batch_size = 32, epochs = 10)
    
    #モデルの保存
    model.save('./digit_recog_cnn_epoch10.h5')
    
    return model
    

def model_predict(model, X):
    test_y = model.predict(X)
    return test_y

if __name__ == "__main__":
    main()