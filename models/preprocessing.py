"@author: NavinKumarMNK"
import cv2 
import torchvision.transforms as T
import numpy as np

class ImagePreProcessing():
    def __init__(self) -> None:
        pass

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def hist_equalization(image):
        return cv2.equalizeHist(image)
    
    def hist_stretching(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Perform CLAHE (Contrast Limited Adaptive Histogram Equalization)
    def clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def gaussian_blur(self, image, kernel_size=(5,5)):
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def median_blur(self, image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)
  
    def bilateral_filter(self, image, d=5, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    
    def erosion(self, image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    
    def dilation(self, image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
    
    def morph_gradient(self, image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    def background_subtraction(self, image, background):
        return cv2.absdiff(background, image)

    def frame_differencing(self, image, prev_frame):
        return cv2.absdiff(prev_frame, image)
        
    def optical_flow(self, image, prev_frame):
        flow = cv2.calcOpticalFlowFarneback(prev_frame, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def preprocess(self, X):
        return T.Compose([
            T.Lambda(lambda x: self.background_subtraction(x)),
            T.Lambda(lambda x: self.frame_differencing(x)),
            T.Lambda(lambda x: self.optical_flow(x))
            ])

    # transform 
    def transforms(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=30),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def augumentation(self):
        return T.Compose([
            T.RandomAffine(scale=(0.8, 1.2)),
            T.RandomAffine(translate=(0.2, 0.2)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ])


if __name__ == '__main__':
    pass
