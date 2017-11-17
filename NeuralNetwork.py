import os
import numpy as np
from DataExtractor import Extractor
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):

        features = []
        labels = []

        DIR_PATH = r"C:\\Users\\Shpoozipoo\\Desktop\\Hands\\ABC" # The path to the directory containing the images
        de = Extractor(DIR_PATH)
        files = os.listdir(DIR_PATH) # A list of all the files in the directory
        print("Length of pictures is ", len(files))
        cnt = 0
        for file in files:
            # Get the features and label from processing the image.
            (f, l) = de.extract(file)
            features.append(f)
            labels.append(l)
            cnt += 1
        print("processed " + str(cnt))

        d = {}
        for lbl in np.array(labels).tolist():
            if lbl in d:
                d[lbl] += 1
            else:
                d[lbl] = 0
        print(str(d))

        features = np.array(features)
        labels = np.array(labels)

        # Find the unique numbers from the train labels
        classes = np.unique(labels)
        nClasses = len(classes)
        print('Total number of outputs : ', nClasses)
        print('Output classes : ', classes)

        train_features, test_features = features[:len(features) - 10], features[len(features) - 10:]
        train_labels, test_labels = labels[:len(labels) - 10], labels[len(labels) - 10:]

        train_labels_one_hot = to_categorical(train_labels)
        test_labels_one_hot = to_categorical(test_labels)

        # Display the change for category label using one-hot encoding
        print("Original label "+ str(len(train_labels) - 1) + " + : ", train_labels[len(train_labels) - 1])
        print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[len(train_labels) - 1])

        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        model = Sequential()
        model.add(Dense(512, activation="relu", input_shape=(np.prod((100, 100)), )))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(nClasses, activation="softmax"))


        # Dont know what this means or how it works
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        # Ask if user wants to load weights
        answer = str(input("Do you want to load the weights from the previous training session? \n"))
        if answer[0].lower() == "y":
            try:
                model.load_weights(r"C:\\Users\\Shpoozipoo\\Desktop\\Weights\\Sign_Language_Model_2.h5")
            except Exception as e:
                print(str(e))
                print("Too bad. You can't.")
                answer = "no"
        if answer[0].lower() =="n":
            history = model.fit(train_features, train_labels_one_hot, batch_size=32, epochs=32, verbose=1, shuffle=True,
                           ) # validation_data=(test_features, test_labels_one_hot)
            model.save_weights(r"C:\\Users\\Shpoozipoo\\Desktop\\Weights\\Sign_Language_Model_2.h5")

            # [test_loss, test_acc] = model.evaluate(test_features, test_labels_one_hot)
            # print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

            self.model = model

            # Plot the Loss Curves
            # plt.figure(figsize=[8, 6])
            # plt.plot(history.history['loss'], 'r', linewidth=3.0)
            # plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
            # plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
            # plt.xlabel('Epochs ', fontsize=16)
            # plt.ylabel('Loss', fontsize=16)
            # plt.title('Loss Curves', fontsize=16)
            #
            # # Plot the Accuracy Curves
            # plt.figure(figsize=[8, 6])
            # plt.plot(history.history['acc'], 'r', linewidth=3.0)
            # plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
            # plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
            # plt.xlabel('Epochs ', fontsize=16)
            # plt.ylabel('Accuracy', fontsize=16)
            # plt.title('Accuracy Curves', fontsize=16)

            # plt.show()

    def predict(self, features):
        af = np.array([features])
        return self.model.predict_classes(af[[0], :])


# nn = NeuralNetwork()