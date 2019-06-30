import nibabel as nib
import cv2
import numpy as np
import os
import re
import csv
import constant_model

class DataPreprocess(object):
    def __init__(self, roi_x, roi_y, result_z):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.result_z = result_z

    def get_center_point(self, csv_path='./train_center_radii.csv'):
        result = {}
        with open(csv_path, 'r') as f:
            f_csv = csv.reader(f)
            for item in f_csv:
                if f_csv.line_num == 1:
                    continue
                key = re.findall('\d+', item[0])
                result[key[3]] = item[1] + item[2]
            f.close()
        return result

    def get_roi_image(self, from_path_lge, from_path_c0, center_point_csv_path, start_num, end_num):
        center_points = self.get_center_point(csv_path=center_point_csv_path)
        result = np.zeros([end_num - start_num, self.result_z, self.roi_x, self.roi_y])

        for i in range(start_num, end_num):
            case = 'patient' + str(i) + '_LGE.nii.gz'
            current_path = os.path.join(from_path_lge, case)
            center_str_val = center_points[re.findall('\d+', case)[0]]
            center_and_radii = re.findall('\d+', center_str_val)

            nimg = nib.load(current_path)
            img = nimg.get_data()
            t0_file_name = case.replace('_LGE','_C0')
            t0_nimg = nib.load(os.path.join(from_path_c0, t0_file_name))
            t0_img = t0_nimg.get_data()
            for z in range(img.shape[2]):
                center_point_x, center_point_y = int(center_and_radii[0]), int(center_and_radii[1])
                center_point_x = center_point_x * (img.shape[0]/t0_img.shape[0])
                center_point_y = center_point_y * (img.shape[1]/t0_img.shape[1])

                x_left = int(center_point_x-self.roi_x/2)
                x_right = int(center_point_x+self.roi_x/2)
                y_left = int(center_point_y-self.roi_y/2)
                y_right = int(center_point_y+self.roi_y/2)
                if x_left < 0: x_left = 0
                if x_right > img.shape[0]: x_right = img.shape[0]
                if y_left < 0: y_left = 0
                if y_right > img.shape[1]: y_right = img.shape[1]
                roi_image = img[x_left:x_right, y_left:y_right, z]

                if roi_image.shape[0] < self.roi_x or roi_image.shape[1] < self.roi_y:
                    roi_image = cv2.copyMakeBorder(roi_image, self.roi_x-roi_image.shape[0], 0, self.roi_y-roi_image.shape[1], 0, cv2.BORDER_CONSTANT, value=0)
                norm = np.uint8(cv2.normalize(roi_image, None, 0, 255, cv2.NORM_MINMAX))
                norm = cv2.equalizeHist(norm)
                if norm.shape[0] < self.roi_x or norm.shape[1] < self.roi_y:
                    norm = cv2.copyMakeBorder(norm, self.roi_x-norm.shape[0], 0, self.roi_y-norm.shape[1], 0, cv2.BORDER_CONSTANT, value=0)
                result[i-start_num, z, ::, ::] = norm
        return result/255

    def get_roi_label(self, from_path_manual, from_path_c0, center_point_csv_path, start_num, end_num):
        center_points = self.get_center_point(csv_path=center_point_csv_path)
        result = np.zeros([end_num - start_num, self.result_z, self.roi_x, self.roi_y])

        for i in range(start_num, end_num):
            case = 'patient' + str(i) + '_LGE_manual.nii.gz'
            current_path = os.path.join(from_path_manual, case)
            center_str_val = center_points[re.findall('\d+', case)[0]]
            center_and_radii = re.findall('\d+', center_str_val)

            nimg = nib.load(current_path)
            img = nimg.get_data()
            t0_file_name = case.replace('_LGE_manual','_C0')
            t0_nimg = nib.load(os.path.join(from_path_c0, t0_file_name))
            t0_img = t0_nimg.get_data()
            for z in range(img.shape[2]):

                center_point_x, center_point_y = int(center_and_radii[0]), int(center_and_radii[1])
                center_point_x = center_point_x * (img.shape[0]/t0_img.shape[0])
                center_point_y = center_point_y * (img.shape[1]/t0_img.shape[1])

                x_left = int(center_point_x-self.roi_x/2)
                x_right = int(center_point_x+self.roi_x/2)
                y_left = int(center_point_y-self.roi_y/2)
                y_right = int(center_point_y+self.roi_y/2)
                if x_left < 0: x_left = 0
                if x_right > img.shape[0]: x_right = img.shape[0]
                if y_left < 0: y_left = 0
                if y_right > img.shape[1]: y_right = img.shape[1]
                roi_label = img[x_left:x_right, y_left:y_right, z]

                if roi_label.shape[0] < self.roi_x or roi_label.shape[1] < self.roi_y:
                    roi_label = cv2.copyMakeBorder(roi_label, self.roi_x-roi_label.shape[0], 0, self.roi_y-roi_label.shape[1], 0, cv2.BORDER_CONSTANT, value=0)
                roi_label[roi_label == 600] = 3
                roi_label[roi_label == 500] = 2
                roi_label[roi_label == 200] = 1
                result[i-start_num, z, :, :] = roi_label
        return result

    def save_image(self, images, pre_name, save_path):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                save_name = os.path.join(save_path, pre_name + str(int(i+1)).zfill(3) + '_' + str(j+1).zfill(2) + '_.png')
                cv2.imwrite(save_name, images[i, j, :, :]*255)
                print(save_name)
            # return


if __name__ == '__main__':
    data_preprocess = DataPreprocess(roi_x=256, roi_y=256, result_z=20)
    train_image = data_preprocess.get_roi_image(from_path_lge='../../data/c0t2lge', from_path_c0='../../data/c0t2lge', center_point_csv_path='../c0/center_radii.csv', start_num=1, end_num=6)
    train_label = data_preprocess.get_roi_label(from_path_manual='../../data/lgegt', from_path_c0='../../data/c0t2lge', center_point_csv_path='../c0/center_radii.csv', start_num=1, end_num=6)
    test_image = data_preprocess.get_roi_image(from_path_lge='../../data/c0t2lge', from_path_c0='../../data/c0t2lge', center_point_csv_path='../c0/center_radii.csv', start_num=6, end_num=46)
    print(train_image.shape)
    print(train_label.shape)
    print(test_image.shape)
    print(os.getcwd())
    data_preprocess.save_image(train_image, 'train_image', './train_image')
    data_preprocess.save_image(train_label, 'train_label', './train_label')
    data_preprocess.save_image(test_image, 'test_image', './test_image')



