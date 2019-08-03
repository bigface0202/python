from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
#画像サイズを小さくするために50pxに縮小する
image_size = 50

#画像の読み込み

X = []
Y = []
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")#パターン一致でファイル名を取得
    for i, file in enumerate(files):
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))#縦横50px
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

#分割処理（３：１）
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
#4つのデータを１つに保存
xy = (X_train, X_test, y_train, y_test)
#numpyの配列をテキストファイルで保存
np.save("./animal.npy", xy)
