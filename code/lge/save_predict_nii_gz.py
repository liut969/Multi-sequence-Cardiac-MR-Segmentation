import nibabel as nib
import os
import re
import cv2
import numpy as np
import csv
import constant_model

def get_center_point(csv_path=constant_model.center_point_csv_path):
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

def save_nii_gz_data(from_path, f_name, predict_png_result_path, num, predict_nii_gz_result_path, center_and_radii):
    center_point_x, center_point_y = int(center_and_radii[0]), int(center_and_radii[1])
    result = []
    save_name = ''
    nimg = nib.load(os.path.join(from_path, f_name))
    img = nimg.get_data()
    t0_file_name = f_name.replace('_LGE','_C0')
    t0_nimg = nib.load(os.path.join('../../data/c0t2lge', t0_file_name))
    t0_img = t0_nimg.get_data()

    center_point_x = int(center_point_x * (img.shape[0]/t0_img.shape[0]))
    center_point_y = int(center_point_y * (img.shape[1]/t0_img.shape[1]))

    for i in range(img.shape[2]):
        result_file_name = os.path.join(predict_png_result_path, str(int(num[0])-5).zfill(3) + '_' + str(i+1).zfill(2) + '.png')
        current_img = cv2.imread(result_file_name)
        current_array = np.array(current_img[:, :, 0])

        current_size = 256
        top = int(center_point_x - current_size/2)
        bottom = int(img.shape[0] - center_point_x - current_size/2)
        left = int(center_point_y - current_size/2)
        right = int(img.shape[1] - center_point_y - current_size/2)
        current_array = cv2.copyMakeBorder(current_array, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        current_array = current_array.T          ### (X, Y)
        current_array = current_array/85
        result.append(current_array)
    result = np.array(result)                    #### (Z, Y, X)
    result = result.T
    result = result.astype(np.int)

    result[result == 1] = 200
    result[result == 2] = 500
    result[result == 3] = 600

    save_name = 'patient' + str(num[0]) + '_LGE_predict.nii.gz'
    save_name = os.path.join(predict_nii_gz_result_path, save_name)
    nimg_save = nib.Nifti1Image(result, affine=nimg.affine, header=nimg.header)
    nimg_save.to_filename(save_name)
    print(save_name)

def save_data_dir(from_path, predict_png_result_path, predict_nii_gz_result_path):
    center_points = get_center_point(csv_path=constant_model.center_point_csv_path)
    # for i in range(constant_model.test_start_num, constant_model.test_end_num):
    for i in range(constant_model.data_test_start_num, constant_model.data_test_end_num):
        f_name = 'patient' + str(i) + '_LGE.nii.gz'
        num = re.findall('\d+', f_name)
        center_str_val = center_points[num[0]]
        center_and_radii = re.findall('\d+', center_str_val)
        save_nii_gz_data(from_path, f_name, predict_png_result_path, num, predict_nii_gz_result_path, center_and_radii)



if __name__ == '__main__':
    from_path = constant_model.get_test_image_from_path
    predict_png_result_path = constant_model.save_path
    predict_nii_gz_result_path = constant_model.predict_nii_gz_result_path
    if not os.path.isdir(predict_nii_gz_result_path):
        os.makedirs(predict_nii_gz_result_path)
    save_data_dir(from_path, predict_png_result_path, predict_nii_gz_result_path)
