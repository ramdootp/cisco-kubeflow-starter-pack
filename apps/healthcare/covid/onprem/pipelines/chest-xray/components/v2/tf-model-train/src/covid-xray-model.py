
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D,AveragePooling3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import numpy as np
import argparse
import cv2
import os


from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
tf.keras.backend.set_learning_phase(0) 
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 32


print("Num GPUs Available: " ,len(tf.config.experimental.list_physical_devices('GPU')))

def main(unused_args):
    
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images("/opt/data/datasets"))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(label)
    
    data = np.array(data) / 255.0
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    
    (trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=.2)
    
    # load the VGG16 network, ensuring the head FC layer sets are left
    # off
    baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False
        
    num_gpu= len(tf.config.experimental.list_physical_devices('GPU'))
    parallel_model=multi_gpu_model(model,gpus=num_gpu)
    parallel_model
    
    parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    history=parallel_model.fit(trainX,trainY,batch_size=BS,epochs=EPOCHS, validation_split=0.1)

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = parallel_model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    # print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

   
    inputs={"image": t for t in parallel_model.inputs}
    outputs={t.name: t for t in parallel_model.outputs}
    MODEL_EXPORT_PATH='/mnt/Model_Covid'
    
    dirFiles = os.listdir(MODEL_EXPORT_PATH)
    if len(dirFiles)>0:
        for i in dirFiles:
            if i=='.ipynb_checkpoints':
                dirFiles.remove(i)
                dirFiles.append(0)
        test_list = [int(i) for i in dirFiles] 
        modelno = max(test_list)+1
    else:
        modelno = 1

    tf.saved_model.simple_save(
        tf.keras.backend.get_session(),
        os.path.join(MODEL_EXPORT_PATH,str(modelno)),
        inputs=inputs,
        outputs={t.name: t for t in parallel_model.outputs})


if __name__ == "__main__":
    tf.app.run()
