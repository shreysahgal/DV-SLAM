import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cos_sim
from tqdm import tqdm

def makeSiftVocab(images, train_size=None, dict_size=200, retries=1):
    
    if not train_size:
        train_size = len(images) // 4

    detector = cv.SIFT_create()
    unclustered = np.zeros((0, 128), dtype=np.float32)

    for img in tqdm(images[:train_size]):
        img = cv.resize(img, (512, 256))
        kp = detector.detect(img, None)
        kp, des = detector.compute(img, kp)
        unclustered = np.vstack((unclustered, des))

    tc = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.001)
    flags = cv.KMEANS_PP_CENTERS

    bow_trainer = cv.BOWKMeansTrainer(dict_size, tc, retries, flags)
    dictionary = bow_trainer.cluster(unclustered)

    return dictionary

def bowVectorizeImages(images, dictionary):
    matcher = cv.FlannBasedMatcher()
    detector = cv.SIFT_create()
    extract = cv.SIFT_create()
    bow_extract = cv.BOWImgDescriptorExtractor(extract, matcher)
    bow_extract.setVocabulary(dictionary)

    bowvecs = list()

    for img in tqdm(images):
        img = cv.resize(img, (512, 256))
        kp = detector.detect(img, None)
        bowvec = bow_extract.compute(img, kp)
        bowvecs.append(bowvec)
    
    return bowvecs

def detectClosures(bowvecs, thold=0.13):
    closures = list()
    for i in tqdm(range(len(bowvecs))):
        for j in range(i+1, len(bowvecs)):
            score = cos_sim(bowvecs[i], bowvecs[j])
            if i != j and score < thold:
                closures.append((i, j))
    return closures

if __name__ == '__main__':

    print('Reading images...')
    img_path = '../leftImg8bit_trainvaltest/leftImg8bit/train/strasbourg'
    filenames = [img for img in glob.glob(img_path + "/*.png")]
    filenames.sort()

    images = []
    for fname in tqdm(filenames):
        img = cv.imread(fname)
        images.append(img)

    print('Traing cluster algorithm...')
    dictionary = makeSiftVocab(images, train_size=90, dict_size=200, retries=5)

    print('BoW-vectorizing images...')
    bowvecs = bowVectorizeImages(images, dictionary)

    print('Detecting closures...')
    closures = detectClosures(bowvecs)

    print('Done! Found %d closures.' % len(closures))

    for i, j in closures[:3]:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(images[i])
        ax2.imshow(images[j])
        
    plt.show()