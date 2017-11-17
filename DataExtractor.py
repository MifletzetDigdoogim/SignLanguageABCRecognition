import cv2
import numpy as np

class Extractor:

    def __init__(self, dir_path):
        print("DataExtractor Initialized!")
        self.dir_path = dir_path

    def extract(self, file):
        self.file = file
        label = int(str(self.file).split("-")[0])
        print(str(self.file).split("-")[0] + " --- " + str(self.file).split("-")[1])
        print(str(self.file) + " --- " + str(label))
        features = []
        full_path = self.dir_path + "\\" + self.file
        image = cv2.imread(full_path, 0)
        height, width = image.shape[:2]
        for y in range(height):
            for x in range(width):
                if image[y, x] > 255//2:
                    # White Pixel
                    features.append(1)
                else:
                    # Black pixel
                    features.append(0)
        # cv2.imshow(str(self.file), image)
        return (np.array(features), np.array(label))
    def extract_features(self, image):
        # Image is already read.
        features = []
        height, width = image.shape[:2]
        for y in range(height):
            for x in range(width):
                if image[y, x] > 255 // 2:
                    # White Pixel
                    features.append(1)
                else:
                    # Black pixel
                    features.append(0)
        return np.array(features)