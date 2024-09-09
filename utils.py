import os
import collections.abc
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


def split_dataset(root):

    data_train = sio.loadmat(os.path.join(root, 'traindata.mat'))['traindata'][0]
    data_test = sio.loadmat(os.path.join(root, 'testdata.mat'))['testdata'][0]
    img_train, label_train = zip(*[(x[0][0], x[1][0]) for x in data_train])
    img_test, label_test = zip(*[(x[0][0], x[1][0]) for x in data_test])

    list_img = img_train + img_test
    list_label = label_train + label_test
    x_train, x_test, y_train, y_test = train_test_split(list_img, list_label)

    return x_train, x_test, y_train, y_test


class IIIT5k(Dataset):
    def __init__(self, root='IIIT5K', list_dataset=None, transform=None):
        # data_str = 'traindata' if training else 'testdata'
        # data = sio.loadmat(os.path.join(root, data_str+'.mat'))[data_str][0]
        # self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in data])
        self.root = root
        self.img, self.label = zip(*list_dataset)
        self.transform = transform
        # if transform is not None:
        #     self.transform = transform
        #     self.img = [self.transform(Image.open(root + '/' + img)) for img in self.img]
        # else:
        #     self.img = [Image.open(root + '/' + img) for img in self.img]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_path = self.root + '/' + self.img[idx]
        img = Image.open(img_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

class synthDataset(Dataset):
    def __init__(self, root="90kDICT32px", annotation_file=None, n_file=256, transform=None):
        self.annotation_file = os.path.join(root, annotation_file)
        self.file = open(self.annotation_file, "r")
        self.transform = transform
        self.img_label = []
        self.img_path = []

        for i, f in enumerate(self.file, start=1):
            parts = f.split()
            img, img_uuid = parts[0], parts[1]
            img_path = root + img.replace("./", "/")
            label = img_path.split(img_uuid + '.jpg')[0].split('_')[1]
            self.img_path.append(img_path)
            self.img_label.append(label)
            if i == n_file:
                break

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        label = self.img_label[idx]

        return img, label

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img = img.convert('L')
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        length = 0
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.abc.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
