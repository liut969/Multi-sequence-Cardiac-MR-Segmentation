import nibabel as nib
import os
import re
import cv2
import numpy as np
import constant_model

def save_nii_gz_data(from_path, predict_png_result_path, num, predict_nii_gz_result_path):
    result = []
    save_name = ''
    nimg = nib.load(from_path)
    img = nimg.get_data()
    for i in range(img.shape[2]):
        result_file_name = os.path.join(predict_png_result_path, 'test_image_patient' + str(num[0]) + '_' + str(i+1).zfill(2) + '.png')
        current_img = cv2.imread(result_file_name)
        current_array = np.array(current_img[:, :, 0])
        current_array = current_array.T          ### (X, Y)
        current_array = current_array/85
        result.append(current_array)
    result = np.array(result)                    #### (Z, Y, X)
    result = result.T
    result = result.astype(np.int)

    save_name = 'patient' + str(num[0]) + '_C0_predict.nii.gz'
    save_name = os.path.join(predict_nii_gz_result_path, save_name)
    nimg_save = nib.Nifti1Image(result, affine=nimg.affine, header=nimg.header)
    nimg_save.to_filename(save_name)
    print(save_name)

def save_data_dir(from_path, predict_png_result_path, predict_nii_gz_result_path):
    for i in range(constant_model.test_start_num, constant_model.test_end_num):
        f_name = 'patient' + str(i) + '_C0.nii.gz'
        num = re.findall('\d+', f_name)
        save_nii_gz_data(os.path.join(from_path, f_name), predict_png_result_path, num, predict_nii_gz_result_path)



if __name__ == '__main__':
    from_path = constant_model.get_test_image_from_path
    predict_png_result_path = constant_model.save_path
    predict_nii_gz_result_path = constant_model.predict_nii_gz_result_path
    if not os.path.isdir(predict_nii_gz_result_path):
        os.makedirs(predict_nii_gz_result_path)
    save_data_dir(from_path, predict_png_result_path, predict_nii_gz_result_path)
