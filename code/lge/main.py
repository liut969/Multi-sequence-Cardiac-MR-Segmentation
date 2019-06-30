from model import *
from image_data_generator import *
from data_preprocess import *
import cv2
import os
import numpy as np
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import constant_model
from save_predict_nii_gz import save_data_dir

def train(data_preprocess):
    model = unet_3D(pretrained_weights=None, input_size=constant_model.input_size)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    train_image = data_preprocess.get_roi_image(constant_model.get_train_image_from_path, constant_model.get_train_image_from_path, constant_model.center_point_csv_path, constant_model.data_train_start_num, constant_model.data_train_end_num)
    train_label = data_preprocess.get_roi_label(constant_model.get_train_label_from_path, constant_model.get_train_image_from_path, constant_model.center_point_csv_path, constant_model.data_train_start_num, constant_model.data_train_end_num)
    train_image = train_image[::, ::, ::, ::, np.newaxis]

    batch_size = 1
    params = {
          'dim': (data_preprocess.result_z, data_preprocess.roi_x, data_preprocess.roi_y),
          'batch_size': batch_size,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': False}

    training_generator = DataGenerator(train_image, train_label, **params)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    def scheduler(epoch):
        if epoch % 9 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
        return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)
    model_checkpoint = ModelCheckpoint(constant_model.save_model_name, monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(generator=training_generator, steps_per_epoch=100, epochs=30, callbacks=[model_checkpoint, tensorboard, reduce_lr])

def test(data_preprocess, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model = load_model(constant_model.save_model_name)
    test_image = data_preprocess.get_roi_image(constant_model.get_train_image_from_path, constant_model.get_train_image_from_path, constant_model.center_point_csv_path, constant_model.data_test_start_num, constant_model.data_test_end_num)
    test_image = test_image[::, ::, ::, ::, np.newaxis]
    for patient in range(test_image.shape[0]):
        predict_test = test_image[patient:patient+1, ::, ::, ::, ::]
        new_pos = model.predict([predict_test])
        for count in range(new_pos.shape[1]):
            img = np.zeros((data_preprocess.roi_x, data_preprocess.roi_y))
            img = img.astype('int')
            for current_value in range(4):
                for row in range(data_preprocess.roi_x):
                    for col in range(data_preprocess.roi_y):
                        if new_pos[0, count, row, col, current_value] == max(new_pos[0, count, row, col]):
                            img[row, col] = current_value
            save_name = os.path.join(save_path + str(int(patient+1)).zfill(3) + '_' + str(count+1).zfill(2) + '.png')
            print(patient, count, np.sum(img == 0), np.sum(img == 1), np.sum(img == 2), np.sum(img == 3))
            cv2.imwrite(save_name, img*85)


if __name__ == '__main__':

    data_preprocess = DataPreprocess(roi_x=256, roi_y=256, result_z=20)
    train(data_preprocess)
    test(data_preprocess, constant_model.save_path)

    from_path = constant_model.get_test_image_from_path
    predict_png_result_path = constant_model.save_path
    predict_nii_gz_result_path = constant_model.predict_nii_gz_result_path
    if not os.path.isdir(predict_nii_gz_result_path):
        os.makedirs(predict_nii_gz_result_path)
    save_data_dir(from_path, predict_png_result_path, predict_nii_gz_result_path)

