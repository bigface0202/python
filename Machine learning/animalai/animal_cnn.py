from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D#畳み込みの処理やプーリングなどの処理を行う
from keras.layers import Activation, Dropout, Flatten, Dense #Flatten：データを1次元にする,Dense:全結合層を連結する
from keras.utils import np_utils
import keras
import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
#画像サイズを小さくするために50pxに縮小する
image_size = 50

# メインの関数を定義する
def main():
    X_train, X_test, y_train, y_test = np.load("./animal.npy", allow_pickle=True)
    X_train = X_train.astype("float") / 256.0
    X_test = X_test.astype("float") / 256.0
    y_train = np_utils.to_categorical(y_train, num_classes)#さる，いのしし，からすを[1,0,0], [0,1,0], [0,0,1]に変換
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    model = model_train(X_train, y_train)#訓練データを与えてモデルを学習させる
    model = model_eval(model, X_test, y_test)#モデルとテストデータを与えて評価を行う

def model_train(X, y):
    model = Sequential()
    #32個の各フィルターを3×3, padding:畳み込み結果が同じサイズになるようにピクセルを左右に追加
    #X_trainの中にはデータの数，ピクセル数（X），ピクセル数（Y），（RGB）が入っていて，1つ目はいらないのでX_train.shape[1:]
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = X.shape[1:]))
    #relu：正だけ通して，負は0
    #CNNの回転軸に塗るグリース
    model.add(Activation('relu'))
    model.add(Conv2D(3,3))
    model.add(Activation('relu'))
    #一番大きい値を取り出す
    model.add(MaxPooling2D(pool_size = (2,2)))
    #25%を捨てる
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    #データを1列に並べる
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #最後の出力層のノードは3つ（3種類の画像分類なため）
    model.add(Dense(3))
    #softmax：それぞれの画像の一致度の確率を足し込むと1になる
    model.add(Activation('softmax'))
    
    #最適化処理の定義
    #lr:learning rate, decay:学習率を下げていく
    opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
    
    #metrics:評価の値
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = opt,
                  metrics = ['accuracy'])
    #エポック数上げると性能が良くなるらしい（このNNの場合）
    model.fit(X, y, batch_size = 32, epochs = 100)
    
    #モデルの保存
    model.save('./animal_cnn.h5')
    
    return model

def model_eval(model, X, y):
    #verbose：途中の経過を表示する
    scores = model.evaluate(X, y, verbose=1)
    return print('Test Loss:', scores[0], 'Test Accuracy:', scores[1])

#もしもこのプログラムがPythonから直接呼ばれたらmain()を実行
#ほかは各関数の引用することができる
if __name__ == "__main__":
    main()