import cv2
import numpy as np
import os
import math
from multiprocessing import Pool
from functools import partial

class Dark_channel_prior:
    def __init__(self, input_folder, output_folder, omega=0.95, size=15, r=50, eps=0.0001, tx=0.1):
        self.input_folder = 'C:/Users/nexreal/Desktop/New_Sample/data/ganghwa/imgdata/'
        self.output_folder = 'C:/Users/nexreal/Desktop/New_Sample/data/ganghwa/output/'
        self.omega = omega
        self.size = size
        self.r = r
        self.eps = eps
        self.tx = tx

    def dark_channel(self, im):
        b,g,r = cv2.split(im)
        dc = cv2.min(cv2.min(r,g),b);
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.size,self.size))
        dark = cv2.erode(dc,kernel)
        return dark

    def AtmLight(self, im,dark):
        [h,w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000),1))
        darkvec = dark.reshape(imsz,1);
        imvec = im.reshape(imsz,3);

        indices = darkvec.argsort();
        indices = indices[imsz-numpx::]

        atmsum = np.zeros([1,3])
        for ind in range(1,numpx):
           atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx;
        return A

    def TransmissionEstimate(self, im,A):
        omega = 0.95;
        im3 = np.empty(im.shape,im.dtype);

        for ind in range(0,3):
            im3[:,:,ind] = im[:,:,ind]/A[0,ind]

        transmission = 1 - omega * self.dark_channel(im3);
        return transmission

    def Guidedfilter(self, im, p):
        mean_I = cv2.boxFilter(im,cv2.CV_64F,(self.r,self.r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F,(self.r,self.r))
        mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(self.r,self.r))
        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(self.r,self.r))
        var_I   = mean_II - mean_I*mean_I

        a = cov_Ip/(var_I + self.eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a,cv2.CV_64F,(self.r,self.r))
        mean_b = cv2.boxFilter(b,cv2.CV_64F,(self.r,self.r))

        q = mean_a*im + mean_b
        return q

    def TransmissionRefine(self, im,et):
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        #r = 60;
        #eps = 0.0001;
        t = self.Guidedfilter(gray,et);

        return t

    def Recover(self, im,t,A):
        res = np.empty(im.shape,im.dtype)
        t = cv2.max(t,self.tx);

        for ind in range(0,3):
            res[:,:,ind] = (im[:,:,ind]-A[0,ind])/cv2.max(t,0.1) + A[0,ind]

        return res

    def dehaze(self, im):
        I = im.astype('float64')/255;
        dark = self.dark_channel(I);
        A = self.AtmLight(I,dark);
        te = self.TransmissionEstimate(I,A);
        t = self.TransmissionRefine(im,te);
        J = self.Recover(I,t,A);
        return J

    def process_image(self, image_paths):
        # Get a list of all the image paths in the input folder
        #image_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.jpg')]
        #for path in image_paths:
        im = cv2.imread(image_paths)
        #result = self.dark_channel(im)
        result = self.dehaze(im)
        filename = os.path.basename(image_paths)
        output_path = os.path.join(self.output_folder, filename)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, result)
        #image_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.jpg')]

    def process_images(self):
        image_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.jpg')]

        with Pool() as p:
            p.map(partial(self.process_image), image_paths)

