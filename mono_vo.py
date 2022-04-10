import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


class MonoOdometery(object):
    def __init__(self,
                img_file_path,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)),
                detector=cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
        
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})

        '''

        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))


    def detect(self, img):
        '''Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''

        p0 = self.detector.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self):
        '''
        Used to perform visual odometery.

        Returns:
            R -- Rotation matrix
            denoting location of detected keypoint
        '''

        self.p0 = self.detect(self.old_frame)


        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        self.p1, st, err = cv.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        

        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]


        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        E, _ = cv.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp)
        return R, t


    def process_frame(self, id0, id1):
        '''
        Get pair images

        Arguments:
            id0 -- id of the first pair image
            id1 -- id of the second pair image

        '''

        # self.old_frame = cv.imread(self.file_path +str(id0).zfill(6)+'.png', 0)
        # self.current_frame = cv.imread(self.file_path + str(id1).zfill(6)+'.png', 0)
        self.old_frame = cv.imread(self.file_path + str(id0) + '.jpg', 0)
        self.current_frame = cv.imread(self.file_path + str(id1) + '.jpg', 0)
        # cv.imshow('dst_rt', self.old_frame )
        # cv.imshow('dst_rt', self.current_frame)
        # cv.waitKey()

    def img_to_frame(self, img0, img1):
       '''
       Directly load images to frame

       Arguments:
            img0 -- first pair image
            img1 -- second pair image

       '''
       self.old_frame = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
       self.current_frame = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)