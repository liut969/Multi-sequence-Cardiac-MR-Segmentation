"""
reference:https://github.com/mahendrakhened/Automated-Cardiac-Segmentation-and-Disease-Diagnosis/blob/master/data_preprocess/acdc_data_preparation.py
"""
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle
from scipy.fftpack import fftn, ifftn
import numpy as np
import nibabel as nib
import csv
import os
import shutil

class ROI(object):
    def __init__(self, from_path,
                 save_path,
                 start_num,
                 end_num):
        self.from_path = from_path
        self.save_path = save_path
        self.start_num = start_num
        self.end_num = end_num

    def load_nii(self, img_path):
        nimg = nib.load(img_path)
        return nimg.get_data(), nimg.affine, nimg.header
#   Fourier-Hough Transform Based ROI Extraction
    def extract_roi_fft(self, data4D, pixel_spacing, minradius_mm=15, maxradius_mm=45, kernel_width=5, center_margin=8, num_peaks=10, num_circles=20, radstep=2):
        """
        Returns center and radii of ROI region in (i,j) format
        """
        # Data shape:
        # radius of the smallest and largest circles in mm estimated from the train set
        # convert to pixel counts

        pixel_spacing_X, pixel_spacing_Y, _ = pixel_spacing
        minradius = int(minradius_mm / pixel_spacing_X)
        maxradius = int(maxradius_mm / pixel_spacing_Y)

        ximagesize = data4D.shape[0]
        yimagesize = data4D.shape[1]
        zslices = data4D.shape[2]
        tframes = 1
        xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
        ysurface = np.tile(range(yimagesize), (ximagesize, 1))
        lsurface = np.zeros((ximagesize, yimagesize))

        allcenters = []
        allaccums = []
        allradii = []

        data4D = data4D[::, ::, ::, np.newaxis]
        for slice in range(zslices):
            ff1 = fftn([data4D[:,:,slice, t] for t in range(tframes)])
            fh = np.absolute(ifftn(ff1[0, :, :]))
            fh[fh < 0.1 * np.max(fh)] = 0.0
            image = 1. * fh / np.max(fh)
            # find hough circles and detect two radii
            edges = canny(image, sigma=3)
            hough_radii = np.arange(minradius, maxradius, radstep)
            # print hough_radii
            hough_res = hough_circle(edges, hough_radii)
            if hough_res.any():
                centers = []
                accums = []
                radii = []

                for radius, h in zip(hough_radii, hough_res):
                    # For each radius, extract num_peaks circles
                    peaks = peak_local_max(h, num_peaks=num_peaks)
                    centers.extend(peaks)
                    accums.extend(h[peaks[:, 0], peaks[:, 1]])
                    radii.extend([radius] * num_peaks)

                # Keep the most prominent num_circles circles
                sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

                for idx in sorted_circles_idxs:
                    center_x, center_y = centers[idx]
                    allcenters.append(centers[idx])
                    allradii.append(radii[idx])
                    allaccums.append(accums[idx])
                    brightness = accums[idx]
                    lsurface = lsurface + brightness * np.exp(
                        -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

        lsurface = lsurface / lsurface.max()
        # select most likely ROI center
        roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

        # determine ROI radius
        roi_x_radius = 0
        roi_y_radius = 0
        for idx in range(len(allcenters)):
            xshift = np.abs(allcenters[idx][0] - roi_center[0])
            yshift = np.abs(allcenters[idx][1] - roi_center[1])
            if (xshift <= center_margin) & (yshift <= center_margin):
                roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
                roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

        if roi_x_radius > 0 and roi_y_radius > 0:
            roi_radii = roi_x_radius, roi_y_radius
        else:
            roi_radii = None

        return roi_center, roi_radii

    #   Stddev-Hough Transform Based ROI Extraction
    def extract_roi_stddev(self, data4D, pixel_spacing, minradius_mm=15, maxradius_mm=45, kernel_width=5,  center_margin=8, num_peaks=10, num_circles=20, radstep=2):
        """
        Returns center and radii of ROI region in (i,j) format
        """
        # Data shape:
        # radius of the smallest and largest circles in mm estimated from the train set
        # convert to pixel counts
        pixel_spacing_X, pixel_spacing_Y, _, _ = pixel_spacing
        minradius = int(minradius_mm / pixel_spacing_X)
        maxradius = int(maxradius_mm / pixel_spacing_Y)

        ximagesize = data4D.shape[0]
        yimagesize = data4D.shape[1]
        zslices = data4D.shape[2]
        tframes = data4D.shape[3]
        xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
        ysurface = np.tile(range(yimagesize), (ximagesize, 1))
        lsurface = np.zeros((ximagesize, yimagesize))

        allcenters = []
        allaccums = []
        allradii = []

        for slice in range(zslices):
            ff1 = np.array([data4D[:,:,slice, t] for t in range(tframes)])
            fh = np.std(ff1, axis=0)
            fh[fh < 0.1 * np.max(fh)] = 0.0
            image = 1. * fh / np.max(fh)
            # find hough circles and detect two radii
            edges = canny(image, sigma=3)
            hough_radii = np.arange(minradius, maxradius, radstep)
            # print hough_radii
            hough_res = hough_circle(edges, hough_radii)
            if hough_res.any():
                centers = []
                accums = []
                radii = []
                for radius, h in zip(hough_radii, hough_res):
                    # For each radius, extract num_peaks circles
                    peaks = peak_local_max(h, num_peaks=num_peaks)
                    centers.extend(peaks)
                    accums.extend(h[peaks[:, 0], peaks[:, 1]])
                    radii.extend([radius] * num_peaks)

                # Keep the most prominent num_circles circles
                sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

                for idx in sorted_circles_idxs:
                    center_x, center_y = centers[idx]
                    allcenters.append(centers[idx])
                    allradii.append(radii[idx])
                    allaccums.append(accums[idx])
                    brightness = accums[idx]
                    lsurface = lsurface + brightness * np.exp(
                        -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

        lsurface = lsurface / lsurface.max()
        # select most likely ROI center
        roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

        # determine ROI radius
        roi_x_radius = 0
        roi_y_radius = 0
        for idx in range(len(allcenters)):
            xshift = np.abs(allcenters[idx][0] - roi_center[0])
            yshift = np.abs(allcenters[idx][1] - roi_center[1])
            if (xshift <= center_margin) & (yshift <= center_margin):
                roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
                roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

        if roi_x_radius > 0 and roi_y_radius > 0:
            roi_radii = roi_x_radius, roi_y_radius
        else:
            roi_radii = None

        return roi_center, roi_radii

    def save_csv(self, c0_path='../../data/c0t2lge'):
        center_radii_path = self.save_path
        headers = ['image_path', 'center', 'radii']
        rows = []

        for i in range(self.start_num, self.end_num):
            # image_path_4D = os.path.join(self.from_path, 'patient%d_C0_manual.nii.gz' % (i))
            image_path_4D = os.path.join(self.from_path, 'patient%d_C0_gt.nii.gz' % (i))
            image_4D, _, hdr, = self.load_nii(image_path_4D)
            if i < 36:
                c, r = self.extract_roi_fft(image_4D, hdr.get_zooms())
                print(image_path_4D, c, r)
                rows.append([image_path_4D, c, r])
            else:
                file_name = os.path.join(c0_path, 'patient' + str(i) + '_C0.nii.gz')
                nimg = nib.load(file_name)
                img = nimg.get_data()
                # c, r = self.extract_roi_stddev(image_4D, hdr.get_zooms())
                c, r = self.extract_roi_fft(image_4D, hdr.get_zooms())
                print(image_path_4D, c, r, int(c[0]*img.shape[0]/255.), int(c[1]*img.shape[1]/255.))
                rows.append([image_path_4D, (int(c[0]*img.shape[0]/255.), int(c[1]*img.shape[1]/255.)), r])

        with open(center_radii_path, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(rows)


    def copy_and_rename(self, save_file_name):
        if not os.path.isdir(save_file_name):
            os.mkdir(save_file_name)
        for i in range(1, 36):
            shutil.copyfile(os.path.join('../../data/c0gt', 'patient' + str(i) + '_C0_manual.nii.gz'), os.path.join(save_file_name, 'patient' + str(i) + '_C0_gt.nii.gz'))
        for i in range(36, 46):
            shutil.copyfile(os.path.join('../../data/result/predict_nii_gz_result', 'patient' + str(i) + '_C0_predict.nii.gz'), os.path.join(save_file_name, 'patient' + str(i) + '_C0_gt.nii.gz'))



if __name__ == '__main__':
    roi_train = ROI('../../data/result/c0gt_0_45', './center_radii.csv', 1, 46)
    roi_train.copy_and_rename('../../data/result/c0gt_0_45')
    roi_train.save_csv()

