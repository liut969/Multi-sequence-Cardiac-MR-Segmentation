import model
import data
import constant_model
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from save_predict_nii_gz import save_data_dir
from roi import ROI

data.get_train_and_test()

# # model = load_model(constant_model.save_model_name)

myGenerator = data.trainGenerator(batch_size=constant_model.batch_size,
                                  train_path=constant_model.train_path,
                                  image_folder=constant_model.classes_image,
                                  mask_folder=constant_model.classes_label,
                                  aug_dict=constant_model.data_gen_args,
                                  flag_multi_class=constant_model.flag_multi_class,
                                  num_class=constant_model.num_class,
                                  save_to_dir=constant_model.data_gen_save_to_dir,
                                  target_size=constant_model.target_size)
model = model.unet()
model_checkpoint = ModelCheckpoint(constant_model.save_model_name, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGenerator, steps_per_epoch=constant_model.steps_per_epoch, epochs=constant_model.epochs, callbacks=[model_checkpoint, constant_model.tbCallBack])

testGene = data.testGenerator(constant_model.test_path, num_image=constant_model.test_image_nums, target_size=constant_model.target_size)
results = model.predict_generator(testGene, constant_model.test_image_nums, verbose=1)
data.saveResult(constant_model.save_path, results)


from_path = constant_model.get_test_image_from_path
predict_png_result_path = constant_model.save_path
predict_nii_gz_result_path = constant_model.predict_nii_gz_result_path
if not os.path.isdir(predict_nii_gz_result_path):
    os.makedirs(predict_nii_gz_result_path)
save_data_dir(from_path, predict_png_result_path, predict_nii_gz_result_path)

roi_train = ROI('../../data/result/c0gt_0_45', './center_radii.csv', 1, 46)
roi_train.copy_and_rename('../../data/result/c0gt_0_45')
roi_train.save_csv()
