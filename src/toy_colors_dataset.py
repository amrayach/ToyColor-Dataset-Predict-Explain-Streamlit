import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from src.local_linear_explanation import *


colors = {
    'r': np.array([255, 0, 0], dtype=np.uint8),
    'o': np.array([255, 128, 0], dtype=np.uint8),
    'y': np.array([255, 255, 0], dtype=np.uint8),
    'g': np.array([0, 255, 0], dtype=np.uint8),
    'b': np.array([0, 128, 255], dtype=np.uint8),
    'i': np.array([0, 0, 255], dtype=np.uint8),
    'v': np.array([128, 0, 255], dtype=np.uint8)
}

imglen = 5
imgshape = (imglen, imglen, 3)
topleft = tuple([0, [0]])
topright = tuple([0, [imglen - 1]])
botleft = tuple([imglen - 1, [0]])
botright = tuple([imglen - 1, [imglen - 1]])

ignore_rule1 = np.zeros(imgshape)
for corner in [topleft, topright, botleft, botright]: ignore_rule1[corner] = 1
ignore_rule1 = ignore_rule1.ravel().astype(bool)

ignore_rule2 = np.zeros(imgshape)
# ignore_rule2[[0, [1, 2, 3]]] = 1
ignore_rule2[tuple([0, [1, 2, 3]])] = 1
ignore_rule2 = ignore_rule2.ravel().astype(bool)


def random_color():
    return colors[np.random.choice(['r', 'g', 'b', 'v'])]


def Bern(p):
    return np.random.rand() < p


def any_repeats(row):
    n_unique = len(set(tuple(c) for c in row))
    return n_unique < len(row)


def ensure_class_0_rules_apply(img):
    # Rule 1
    img[topleft] = img[botright]
    img[topright] = img[botright]
    img[botleft] = img[botright]

    # Rule 2
    toprow = img[0]
    while any_repeats(toprow[1:-1]):
        toprow[1 + np.random.choice(imglen - 2)] = random_color()


def ensure_class_1_rules_apply(img):
    # Rule 1
    if Bern(0.5):
        while np.array_equal(img[topright], img[botleft]):
            img[topright] = random_color()
    else:
        while np.array_equal(img[topleft], img[botright]):
            img[topleft] = random_color()

    # Rule 2
    toprow = img[0]
    while not any_repeats(toprow[1:-1]):
        toprow[1 + np.random.choice(imglen - 2)] = random_color()


def generate_image(label):
    image = np.array([[random_color()
                       for _ in range(imglen)]
                      for __ in range(imglen)], dtype=np.uint8)

    # image = torch.tensor([[random_color()
    #                   for _ in range(imglen)]
    #                  for __ in range(imglen)], dtype=torch.uint8)

    if label == 0:
        ensure_class_0_rules_apply(image)
    else:
        ensure_class_1_rules_apply(image)

    # image = torch.tensor(image)

    # plt.imshow(image)
    # plt.show()

    # image = image.permute(2, 0, 1)

    # plt.imshow(image.permute(1, 2, 0))

    # plt.show()

    # print()

    # return image.ravel()
    return image


def largest_mag_2d(input_gradients, cutoff=0.67):
    # return 2d arrays of flattened largest-magnitude elements
    # so we can compare pixel locations on a 2d basis rather than
    # worrying about RGB. 2d arrays have 1s if any of the pixel component
    # gradients in that space are above the cutoff.
    return np.array([((
                              np.abs(e) > cutoff * np.abs(e).max()
                      ).reshape(5, 5, 3).sum(axis=2).ravel() > 0
                      ).reshape(5, 5) for e in input_gradients
                     ]).astype(int)


def fraction_inside_corners(mask1):
    mask2 = mask1.copy()
    mask2[0][0] = 0
    mask2[0][-1] = 0
    mask2[-1][0] = 0
    mask2[-1][-1] = 0
    return 1 - mask2.ravel().sum() / float(mask1.ravel().sum())


def fraction_inside_topmids(mask1):
    mask2 = mask1.copy()
    mask2[0][1] = 0
    mask2[0][2] = 0
    mask2[0][3] = 0
    return 1 - mask2.ravel().sum() / float(mask1.ravel().sum())


def rule1_score(model, X):
    return np.mean([fraction_inside_corners(grad) for grad in largest_mag_2d(model.input_gradients(X))])


def rule2_score(model, X):
    return np.mean([fraction_inside_topmids(grad) for grad in largest_mag_2d(model.input_gradients(X))])


def generate_dataset(N=20000, cachefile='data/toy-colors.npz'):
    if cachefile and os.path.exists(cachefile):
        cache = np.load(cachefile)
        data = tuple([cache[f] for f in sorted(cache.files)])
    else:
        train_y = (np.random.rand(N) < 0.5).astype(np.uint8)
        dev_y = (np.random.rand(N) < 0.5).astype(np.uint8)
        test_y = (np.random.rand(N) < 0.5).astype(np.uint8)
        data = (
            torch.tensor([generate_image(y) for y in train_y]),
            torch.tensor([generate_image(y) for y in dev_y]),
            torch.tensor([generate_image(y) for y in test_y]),
            train_y,
            dev_y,
            test_y)
        if cachefile:
            np.savez(cachefile, *data)

    return data


class ToyColorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.data = images
        self.labels = labels
        self.transform = transform
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.data[idx]  # .astype(np.uint8) #.reshape((5, 5, 3))

        # plt.imshow(image)
        # plt.show()

        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        # if self.transform is not None:
        #    image = self.transform(image)
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()

        # print()

        return image, self.labels[idx]


def generate_pytorch_dataset(N=20000, cachefile='data/toy-colors.npz'):
    # generate Train/Test toycolor datasets
    X_train, X_dev, X_test, y_train, y_dev, y_test = generate_dataset(N, cachefile)

    return ToyColorDataset(X_train, y_train, transform=transforms.ToTensor()), \
           ToyColorDataset(X_dev, y_dev, transform=transforms.ToTensor()), \
           ToyColorDataset(X_test, y_test, transform=transforms.ToTensor())


def generate_pytorch_dataset(args):
    now = datetime.now()
    time_stamp = now.strftime("%Y%m%d-%H%M%S")

    data_gen_status = False
    if not args.getboolean('Data', 'generate_new_data'):
        if os.path.isfile(args.get('Data', 'data_dir') + args.get('Data', 'cached_data_file')):
            X_train, X_dev, X_test, y_train, y_dev, y_test = generate_dataset(
                cachefile=args.get('Data', 'data_dir') + args.get('Data', 'cached_data_file'))
            data_gen_status = True

    if args.getboolean('Data', 'generate_new_data') or (not data_gen_status):
        X_train, X_dev, X_test, y_train, y_dev, y_test = generate_dataset(N=args.getint('Data', 'num_data_per_set'),
                                                                          cachefile=args.get('Data',
                                                                                             'data_dir') + 'toy-colors-5_5_3-' + time_stamp + '.npz')

    return ToyColorDataset(X_train, y_train, transform=transforms.ToTensor()), \
           ToyColorDataset(X_dev, y_dev, transform=transforms.ToTensor()), \
           ToyColorDataset(X_test, y_test, transform=transforms.ToTensor())


def generate_dataloaders(args, train_dataset, dev_dataset, test_dataset):

    num_workers = args.getint('Dataset', 'num_workers')

    train_params = {'batch_size': args.getint('Dataset', 'train_batch_size'),
                    'shuffle': args.getboolean('Dataset', 'train_dev_shuffle'),
                    'num_workers': num_workers}
    dev_params = {'batch_size': args.getint('Dataset', 'dev_batch_size'),
                  'shuffle': args.getboolean('Dataset', 'train_dev_shuffle'),
                  'num_workers': num_workers}

    test_params = {'batch_size': args.getint('Dataset', 'test_batch_size'),
                   'shuffle': False,
                   'num_workers': num_workers}

    return DataLoader(train_dataset, **train_params), DataLoader(dev_dataset, **dev_params), DataLoader(test_dataset, **test_params)




if __name__ == '__main__':
    train_dataset, dev_dataset, test_dataset = generate_pytorch_dataset(
        cachefile='../Data/toy_colors/toy-colors_5_5_3.npz')

    train_params = {'batch_size': 1,
                    'shuffle': True,
                    'num_workers': 6}
    dev_params = {'batch_size': 64,
                  'shuffle': True,
                  'num_workers': 6}

    test_params = {'batch_size': 64,
                   'shuffle': False,
                   'num_workers': 6}

    train_iterator = DataLoader(train_dataset, **train_params)
    dev_iterator = DataLoader(dev_dataset, **dev_params)
    test_iterator = DataLoader(test_dataset, **test_params)

    """

    X1 = np.array([])
    Y1 = np.array([])
    for (x, y) in train_iterator:
        x = x.numpy()
        y = y.numpy()
        plt.subplot(121)
        plt.title('Class 1')
        class_1 = x[np.argwhere(y == 0)[:9]]
        image_grid(x[np.argwhere(y == 0)[:9]], (5, 5, 3), 3)
        plt.subplot(122)
        plt.title('Class 2')
        image_grid(x[np.argwhere(y == 1)[:9]], (5, 5, 3), 3)
        plt.show()
        print()
        #X1.extend(x)
        #Y1.extend(y)

    plt.subplot(121)
    plt.title('Class 1')
    class_1 = X[np.argwhere(Y == 0)[:9]]
    image_grid(X[np.argwhere(Y == 0)[:9]], (5, 5, 3), 3)
    plt.subplot(122)
    plt.title('Class 2')
    image_grid(X[np.argwhere(Y == 1)[:9]], (5, 5, 3), 3)
    plt.show()
    print()
    """

    # dataiter = iter(train_iterator)
    # images1, labels1 = dataiter.next()
    # images1 = images1.numpy()[0]
    # plt.imshow(images1)
    # plt.show()
    # images2, labels2 = dataiter.next()
    # images3, labels3 = dataiter.next()

    X, Xa, Xy, y, ya, yt = generate_dataset(cachefile='../Data/toy_colors/toy-colors_5_5_3.npz')

    # exit()
    # train_set, dev_set, test_set = generate_pytorch_dataset(cachefile='../Data/toy_colors/toy-colors.npz')

    # dataiter = iter(train_set)
    # images, labels = dataiter.next()

    plt.subplot(121)
    plt.title('Class 1')
    class_1 = X[np.argwhere(y == 0)[:9]]
    image_grid(X[np.argwhere(y == 0)[:9]], (5, 5, 3), 3)
    plt.subplot(122)
    plt.title('Class 2')
    image_grid(X[np.argwhere(y == 1)[:9]], (5, 5, 3), 3)
    plt.show()

    print()
    # import pdb

    # X, Xt, y, yt = generate_dataset()
    # pdb.set_trace()
