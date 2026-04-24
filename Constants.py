

#========================== public configure ==========================
IMG_SIZE = (512, 512)
TOTAL_EPOCH = 600
INITAL_EPOCH_LOSS = 1000000
NUM_EARLY_STOP = 60
NUM_UPDATE_LR = 100
BINARY_CLASS = 1
BATCH_SIZE = 2
learning_rates =1e-3


# ===================   DRIVE configure =========================
DATA_SET = 'HRF
visual_samples = '../log/visual_samples/'
saved_path = '../log/weights_save/'+ DATA_SET + '/'
visual_results = '../log/visual_results/'+ DATA_SET + '/'

resize_drive = 512
resize_size_drive = (resize_drive, resize_drive)
size_h, size_w = 584, 565

# 注意！！！
# 1、现在是584*565的图像文件保存在/tempt目录下，对应运行read_DRIVE_crop.py 文件
# 2、如果是512*512的要去掉路径的/tempt，对应运行read_DRIVE.py 文件
# path_image_drive = '../dataset1/npy/DRIVE/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/DRIVE/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/DRIVE/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/DRIVE/tempt/test_label_save.npy'
# path_val_image_drive = '../dataset1/npy/DRIVE/tempt/val_image_save.npy'
# path_val_label_drive = '../dataset1/npy/DRIVE/tempt/val_label_save.npy'



# path_image_drive = '../dataset1/npy/CHASE_DB1/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/CHASE_DB1/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/CHASE_DB1/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/CHASE_DB1/tempt/test_label_save.npy'

#
# path_image_drive = '../dataset1/npy/STARE/tempt/train_image_save.npy'
# path_label_drive = '../dataset1/npy/STARE/tempt/train_label_save.npy'
# path_test_image_drive = '../dataset1/npy/STARE/tempt/test_image_save.npy'
# path_test_label_drive = '../dataset1/npy/STARE/tempt/test_label_save.npy'


total_drive = 40
Classes_drive_color = 20
###########################################################################################