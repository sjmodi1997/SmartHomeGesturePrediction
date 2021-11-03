import cv2
import numpy as np
import os
from handshape_feature_extractor import HandShapeFeatureExtractor
import frameextractor
from scipy import spatial

model = HandShapeFeatureExtractor.get_instance()


def generate_penultimate_layer(input_path_name):
    videos = []
    for fileName in os.listdir(input_path_name):
        if fileName.endswith(".mp4"):
            videos.append(os.path.join(input_path_name, fileName))
    feature_vectors = []
    print("Extracting Frames of " + input_path_name)
    for video in videos:
        frame = frameextractor.frame_extractor(video)
        feature = model.extract_feature(frame)
        feature_vectors.append(feature)
    return feature_vectors


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
training = generate_penultimate_layer("traindata")

# =============================================================================
# Get the penultimate layer for test data (Our Data)
# =============================================================================
testing = generate_penultimate_layer("test")

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
#
feature_label = []
cosine_similarity = []
for i in range(3):
    for test in testing:
        for train in training:
            dist = spatial.distance.cosine(test, train)
            cosine_similarity.append(dist)
        feature_label.append(int(cosine_similarity.index(min(cosine_similarity))) % 17)
        cosine_similarity = []

np.savetxt("Results.csv", feature_label, fmt="%d")
