from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
#画像サイズを小さくするために50pxに縮小する
image_size = 50
num_testdata = 100

#画像の読み込み
X_train = []
X_test = []
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")#パターン一致でファイル名を取得
    for i, file in enumerate(files):
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))#縦横50px
        data = np.asarray(image)
        #num_testdataを満たさなかったらデータを増やす
        
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        
        else:
            for angle in range(-20, 20, 5):
                #回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)
                
                #反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

# X = np.array(X)
# Y = np.array(Y)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

#分割処理（３：１）
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
#4つのデータを１つに保存
xy = (X_train, X_test, y_train, y_test)
#numpyの配列をテキストファイルで保存
np.save("./animal_aug.npy", xy)