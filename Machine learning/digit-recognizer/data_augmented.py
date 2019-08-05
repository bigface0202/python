from PIL import Image
import os, glob
import numpy as np
import pandas as pd

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

#pillowで読み込みたいので一旦int32型にする
X_train = train_data.iloc[:, 1:].values.astype('int32')
y_train = train_data['label'].values.astype('int32')

X_train_imgs = X_train.reshape((-1, 28, 28, 1))

X_test = test_data.values.astype('float32')
X_test_imgs = X_test.reshape((-1, 28, 28, 1)) / 255.

X_train_imgs_new = []
y_train_new = []
X_test = []

for i, X_img in enumerate(X_train_imgs):
    pilImage = Image.fromarray(X_img[:, :, 0])
#     pilImage = pilImage.convert("RGB")
    for angle in range(-20, 20, 5):
        #ノーマル
        data = np.asarray(pilImage)
        X_train_imgs_new.append(data)
        y_train_new.append(y_train[i])
        
        #回転
        X_img_r = pilImage.rotate(angle)
        data = np.asarray(X_img_r)
        X_train_imgs_new.append(data)
        y_train_new.append(y_train[i])
                
        #反転
        X_img_trans = pilImage.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(X_img_trans)
        X_train_imgs_new.append(data)
        y_train_new.append(y_train[i])
        
X_train_imgs_new = np.array(X_train_imgs_new)
X_train_imgs_new = X_train_imgs_new.reshape(1008000, 28, 28, 1)
y_train_new = np.array(y_train_new)

print("-----Data shape-----")
print("X_train:", X_train_imgs_new.shape, "Y_train:", y_train_new.shape)
print("--------------------")

#npyとして保存するときにはx_train, x_test, y_train, y_testという風に並べる必要があるみたい
#中身としてはx_train(1008000, 28, 28, 3), x_test(28000, 28, 28, 1), 
#y_train(1008000,), y_test(28000,)
#という感じで並べるとnpyデータを作ることができる
#どれか２セットとかだとデータサイズが違うといったエラーがでる
empty_data_y = np.zeros(28000)

xy = (X_train, X_test_imgs, y_train, empty_data_y)
#numpyの配列をテキストファイルで保存
np.save("./digit_aug_fromVSCode.npy", xy)