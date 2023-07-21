import cv2
import numpy as np
import math
import os

class HazeRemover:
    def __init__(self, input_folder, output_folder, omega=0.95, size=15):
        self.input_folder = 'C:/Users/nexreal/Desktop/New_Sample/data/ganghwa/imgdata/'
        self.output_folder = 'C:/Users/nexreal/Desktop/New_Sample/data/ganghwa/output/'
        self.omega = omega
        self.size = size


    def dark_channel(self, img):
        r, g, b = cv2.split(img)
        min_img = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.size, self.size))
        dark = cv2.erode(min_img,kernel)
        return dark

    def atmospheric_light(self, img, dark):
        [h,w] = img.shape[:2]
        img_size = h*w
        num_pixels = int(max(math.floor(img_size/1000),1))
        dark_vec = dark.reshape(img_size,1)
        img_vec = img.reshape(img_size,3)

        indices = dark_vec.argsort()
        indices = indices[img_size-num_pixels::]

        atmsum = np.zeros([1,3])
        for ind in range(1,num_pixels):
            atmsum = atmsum + img_vec[indices[ind]]

        A = atmsum / num_pixels
        return A

    def transmission_estimate(self, img, A):
        norm_img = img/A
        t = 1 - self.omega*self.dark_channel(norm_img)
        return t

    def haze_removal(self, image_paths):
        img = cv2.imread(image_paths)
        if img is None:
            print(f"Image {image_path} not loaded properly. Check the file path.")
            return

        dark = self.dark_channel(img)
        A = self.atmospheric_light(img, dark)
        te = self.transmission_estimate(img, A)
        img = np.float64(img)/255
        t = np.float64(te)
        t = cv2.max(t,0.1)

        J = np.empty(img.shape, img.dtype)
        for ind in range(0,3):
            J[:,:,ind] = (img[:,:,ind]-A[0,ind])/t + A[0,ind]

        output_path = os.path.join(self.output_folder, os.path.basename(image_paths))
        cv2.imwrite(output_path, J*255)

    def process_images(self):
        # Get a list of all the image paths in the input folder
        image_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.jpg')]
        #image_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.jpg')]

        for image_path in image_paths:
            self.haze_removal(image_path)

#HazeRemover.haze_removal("input.jpg", "output_folder")