import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class DogsCats(data.Dataset):

    def __init__(self, filepath, train=True, test=False):
        """
        Generate the train, test, valid dataset.

        :param filepath: image path
        :param train: ture is train mode; false and test=false is valid mode.
        :param test: true is test mode.
        """
        self.isTrain = train
        self.isTest = test

        # load images from filepath
        if self.isTest and not self.isTrain:  # only test mode
            imgs = [os.path.join(filepath + '/test', img) for img in os.listdir(filepath + '/test')]
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:  # train or valid mode
            imgs = [os.path.join(filepath+'/train', img) for img in os.listdir(filepath+'/train')]
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        if self.isTest:
            self.imgs = imgs
        elif self.isTrain:
            self.imgs = imgs[:int(0.8 * imgs_num)]  # train: valid = 8: 2
        else:
            self.imgs = imgs[int(0.8 * imgs_num):]

        # normalize each channel of the input by (input[channel] = (input[channel] - mean[channel]) / std[channel])
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        if self.isTest or not self.isTrain:  # for test image
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
        else:  # for train or valid image
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        """
        Return a image(and corresponding label) as tensor type.
        :param index:
        :return:
        """
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = self.transforms(img)

        if self.isTest:
            id = int(img_path.split('.')[-2].split('/')[-1])
            return img, id
        else:
            label = 0 if 'cat' in img_path.split('/')[-1] else 1
            return img, label

    def __len__(self):
        """
        Length of dataset.
        :return: length of dataset.
        """
        return len(self.imgs)
