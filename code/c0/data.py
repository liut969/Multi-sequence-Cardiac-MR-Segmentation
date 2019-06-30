from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import nibabel as nib
import re
import cv2
from keras.utils import to_categorical
import constant_model

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask=np.reshape(mask,[mask.shape[0], mask.shape[1]*mask.shape[2]])
        new_mask = to_categorical(new_mask, num_classes=num_class, dtype='uint8')
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)



def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=constant_model.flag_multi_class, num_class=constant_model.num_class,
                   save_to_dir=None, target_size=constant_model.target_size, seed=1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

def testGenerator(test_path, num_image, target_size = constant_model.target_size, flag_multi_class = constant_model.flag_multi_class, as_gray = True):
    for f_name in [f for f in os.listdir(test_path) if f.endswith('.png')]:
        img = io.imread(os.path.join(test_path, f_name), as_gray=as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img


def saveResult(save_path, npyfile, flag_multi_class = constant_model.flag_multi_class, num_class = constant_model.num_class):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    f_names = []
    for f_name in [f for f in os.listdir(constant_model.get_test_image_save_path) if f.endswith('.png')]:
        f_names.append(f_name)
    for i, item in enumerate(npyfile):
        item = np.reshape(item, constant_model.target_size+(item.shape[1],))
        img = np.zeros((item.shape[0], item.shape[1]))
        img = img.astype('int')
        for current_value in range(item.shape[2]):
            for row in range(item.shape[0]):
                for col in range(item.shape[1]):
                    if item[row, col, current_value] == max(item[row, col]):
                        img[row, col] = current_value
        print(i, f_names[i], np.sum(img == 0), np.sum(img == 1), np.sum(img == 2), np.sum(img == 3), np.sum(item[:, :, 1] != 0))
        test_image = io.imread(os.path.join(constant_model.get_test_image_save_path, f_names[i]))
        img = cv2.resize(img, (test_image.shape[1], test_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        io.imsave(os.path.join(save_path, f_names[i]), img*85)


def get_image(from_path, save_path, num, pre_name):
    print(from_path)
    nimg = nib.load(from_path)
    img = nimg.get_data()
    for i in range(img.shape[2]):
        save_name = os.path.join(save_path, pre_name + str(num[0]) + '_' + str(i+1).zfill(2) + '.png')
        norm = img[:, :, i]
        norm = np.uint8(cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX))
        norm = cv2.equalizeHist(norm)
        norm = cv2.resize(norm, constant_model.target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_name, norm)

def get_label(from_path, save_path, num, pre_name):
    print(from_path)
    nimg = nib.load(from_path)
    img = nimg.get_data()
    img[img == 600] = 3
    img[img == 500] = 2
    img[img == 200] = 1
    for i in range(img.shape[2]):
        save_name = os.path.join(save_path, pre_name + str(num[0]) + '_' + str(i+1).zfill(2) + '.png')
        img = cv2.resize(img, constant_model.target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_name, img[:, :, i])

def get_image_dir(from_path, save_path, start_num, end_num, pre_name):
    for i in range(start_num, end_num):
        f_name = 'patient' + str(i) + '_C0.nii.gz'
        num = re.findall('\d+', f_name)
        get_image(os.path.join(from_path, f_name), save_path, num, pre_name)

def get_label_dir(from_path, save_path, start_num, end_num, pre_name):
    for i in range(start_num, end_num):
        f_name = 'patient' + str(i) + '_C0_manual.nii.gz'
        num = re.findall('\d+', f_name)
        get_label(os.path.join(from_path, f_name), save_path, num, pre_name)

def get_train_and_test():
    from_path = constant_model.get_train_image_from_path
    save_path = constant_model.get_train_image_save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    get_image_dir(from_path, save_path, constant_model.train_start_num, constant_model.train_end_num, 'train_image_patient')

    from_path = constant_model.get_train_label_from_path
    save_path = constant_model.get_train_label_save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    get_label_dir(from_path, save_path, constant_model.train_start_num, constant_model.train_end_num, 'train_label_patient')

    from_path = constant_model.get_test_image_from_path
    save_path = constant_model.get_test_image_save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    get_image_dir(from_path, save_path, constant_model.test_start_num, constant_model.test_end_num, 'test_image_patient')

    # from_path = constant_model.get_test_label_from_path
    # save_path = constant_model.get_test_label_save_path
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # get_label_dir(from_path, save_path, 81, 101, 'test_label_patient')

if __name__ == '__main__':
    myGenerator = trainGenerator(batch_size=constant_model.batch_size,
                                  train_path=constant_model.train_path,
                                  image_folder=constant_model.classes_image,
                                  mask_folder=constant_model.classes_label,
                                  aug_dict=constant_model.data_gen_args,
                                  flag_multi_class=constant_model.flag_multi_class,
                                  num_class=constant_model.num_class,
                                  save_to_dir=constant_model.data_gen_save_to_dir,
                                  target_size=constant_model.target_size)
    for batch in enumerate(myGenerator):
        print(batch)
        break
