import cv2
import numpy as np
import os
from handshape_feature_extractor import HandShapeFeatureExtractor
import frameextractor
from scipy import spatial
model = HandShapeFeatureExtractor.get_instance()


def generatePenultimateLayer(inputPathName):
    videos = []
    for fileName in os.listdir(inputPathName):
        if fileName.endswith(".mp4"):
            videos.append(os.path.join(inputPathName, fileName))
    featureVectors = []
    print("Extracting Frames of " + inputPathName)
    for video in videos:
        frame = frameextractor.frame_extractor(video)
        feature = model.extract_feature(frame)
        featureVectors.append(feature)
    return featureVectors

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
training = generatePenultimateLayer("test")

# =============================================================================
# Get the penultimate layer for test data (Our Data)
# =============================================================================
testing = generatePenultimateLayer("test")

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
#
featureLabel = []
cosineSimilarity = []
for i in range(3):
    for test in testing:
        for train in training:
            dist = spatial.distance.cosine(test, train)
            cosineSimilarity.append(dist)
        featureLabel.append(int(cosineSimilarity.index(min(cosineSimilarity))) % 17)
        cosineSimilarity = []


np.savetxt("Results.csv", featureLabel, fmt="%d")
