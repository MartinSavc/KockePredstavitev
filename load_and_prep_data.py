import matplotlib.pyplot as pyplot
import numpy as np
import scipy.ndimage as ndimage
import os


def load_imgs_from_folder(folder, ext='png'):
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(ext)]

    for f in files:
        name, _ = f.rsplit('.', 1)
        img = pyplot.imread(os.path.join(folder, f))
        if img.dtype==np.uint8:
            img = np.float32(img)/255.
        yield name, img

def load_mask_for_image(folder, name, gauss_sigma=None):
    file_path = os.path.join(folder, 'masks', name+'.png')
    mask = pyplot.imread(file_path)
    if mask.ndim == 3:
        mask = mask[..., 1]
    if gauss_sigma is not None:
        mask = ndimage.gaussian_filter(mask, sigma=gauss_sigma)
    mask = mask/mask.max()
    return mask


def load_and_prep_data(folder):
    imgs = []
    masks = []
    for name, img in load_imgs_from_folder(folder):
        mask = load_mask_for_image(folder, name, 5)

        imgs += [img]
        masks += [mask]

    return imgs, masks

def load_resample_and_prep_data(folder, win_size=128, resamples=1, mask_gauss=5, scale_range=(0.9, 1.1), angle_range=(0, 2*np.pi)):

    for name, img in load_imgs_from_folder(folder):
        mask = load_mask_for_image(folder, name)
        pts_y, pts_x = np.where(mask)

        mask_lp = ndimage.gaussian_filter(mask, sigma=mask_gauss)
        mask_lp = np.float32(mask_lp/mask_lp.max())

        for cx, cy in zip(pts_x, pts_y):

            rand_scale_list = scale_range[0]+np.random.rand(resamples)*(scale_range[1]-scale_range[0])
            rand_angle_list = angle_range[0]+np.random.rand(resamples)*(angle_range[1]-angle_range[0])

            for rand_scale, rand_angle in zip(rand_scale_list, rand_angle_list):
                # move from 0,0 to center of object
                Tr_1 = np.eye(3)
                Tr_1[:2, 2] = cy, cx

                # scale are around object
                S = np.eye(3)
                S[:2,:2] *= rand_scale

                # rotate area around object
                R = np.eye(3)
                R[0, 0] = np.cos(rand_angle)
                R[0, 1] = np.sin(rand_angle)
                R[1, 0] = -1*R[0, 1]
                R[1, 1] = R[0, 0]

                # move center of window to 0,0
                Tr_2 = np.eye(3)
                Tr_2[:2, 2] = -win_size/2

                # inverse transform:
                # move center of window to 0,0
                # rotate and scale
                # move window to center of object
                M_inv = Tr_1.dot(S.dot(R.dot(Tr_2)))

                # apply the transforms to image and filtered mask
                img_sample = np.zeros((win_size, win_size, 3), dtype=np.float32)
                img_sample[..., 0] = ndimage.affine_transform(img[..., 0], M_inv, output_shape=(win_size, win_size))
                img_sample[..., 1] = ndimage.affine_transform(img[..., 1], M_inv, output_shape=(win_size, win_size))
                img_sample[..., 2] = ndimage.affine_transform(img[..., 2], M_inv, output_shape=(win_size, win_size))
                mask_sample = ndimage.affine_transform(mask_lp, M_inv, output_shape=(win_size, win_size))
                yield img_sample, mask_sample

'''***'''
def load_dice_data_from_folder(folder, data_file='label_data.csv'):
    with open(os.path.join(folder, data_file), 'r') as data_file_obj:
        header = data_file_obj.readline()
        prev_img_name = None
        img = None

        for line in data_file_obj.readlines():
            img_name, color, xpos, ypos, value = line.split(' ')
            ypos = int(ypos)
            xpos = int(xpos)
            value = int(value)
            if img_name != prev_img_name:
                img = pyplot.imread(os.path.join(folder, img_name+'.png'))
                if img.dtype == np.uint8:
                    img = np.float32(img)/255.

            yield img, color, (ypos, xpos, value)


def load_resample_and_prep_dice_data_from_folders(folders_list,
                                                  data_files='label_data.csv',
                                                  win_size=31,
                                                  resamples=1,
                                                  scale_range=(0.9, 1.1),
                                                  angle_range=(0, 2*np.pi),
                                                  trans_range=(0, 0)):
    if type(folders_list) is str:
        folders_list = [folders_list]

    if type(data_files) is str:
        data_files = [data_files]*len(folders_list)

    if len(data_files) != len(folders_list):
        raise Exception('length of list of folders ({:})'+\
                        ' and data files ({:}) do not match'.format(folders_list, data_files))

    for folder, data_file in zip(folders_list, data_files):
        for img, color, (cy, cx,  value) in load_dice_data_from_folder(folder, data_file):

            rand_scale_list = scale_range[0]+np.random.rand(resamples)*(scale_range[1]-scale_range[0])
            rand_angle_list = angle_range[0]+np.random.rand(resamples)*(angle_range[1]-angle_range[0])
            rand_trans_list = trans_range[0]+np.random.rand(resamples,2)*(trans_range[1]-trans_range[0])

            for rand_scale, rand_angle, rand_trans_list in zip(rand_scale_list, rand_angle_list, rand_trans_list):
                # move from 0,0 to center of object
                Tr_1 = np.eye(3)
                Tr_1[:2, 2] = cy, cx
                Tr_1[:2, 2] += rand_trans_list

                # scale are around object
                S = np.eye(3)
                S[:2,:2] *= rand_scale

                # rotate area around object
                R = np.eye(3)
                R[0, 0] = np.cos(rand_angle)
                R[0, 1] = np.sin(rand_angle)
                R[1, 0] = -1*R[0, 1]
                R[1, 1] = R[0, 0]

                # move center of window to 0,0
                Tr_2 = np.eye(3)
                Tr_2[:2, 2] = -win_size/2

                # inverse transform:
                # move center of window to 0,0
                # rotate and scale
                # move window to center of object
                M_inv = Tr_1.dot(S.dot(R.dot(Tr_2)))

                # apply the transforms to image and filtered mask
                img_sample = np.zeros((win_size, win_size, 3), dtype=np.float32)
                img_sample[..., 0] = ndimage.affine_transform(img[..., 0], M_inv, output_shape=(win_size, win_size))
                img_sample[..., 1] = ndimage.affine_transform(img[..., 1], M_inv, output_shape=(win_size, win_size))
                img_sample[..., 2] = ndimage.affine_transform(img[..., 2], M_inv, output_shape=(win_size, win_size))
                yield img_sample, color, value 
'''***'''
