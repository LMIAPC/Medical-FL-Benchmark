from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import itertools

def iid_slicing(dataset, num_clients):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    client_sample_nums = int(len(dataset) / (num_clients))
    for i in range(num_clients):
        dict_users[i] = list(np.random.choice(
            all_idxs, client_sample_nums, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

def normalize(img):
    normalized_img = (img - 128.0)/ 255.0
    return normalized_img.astype(np.float32) 
    
def obtainClassNames(root_dir):
    if root_dir.find('IntelImage') >=0:
        CLASS_NAMES = ['buildings','forest','glacier','mountain','sea','street']
    elif root_dir.find('TB') >=0:
        CLASS_NAMES = ['Normal', 'TB']
    elif root_dir.find('ChestX-Ray_Pneumonia') >=0:
        CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
    elif root_dir.find('NCT-CRC-HE-100K') >=0:
        CLASS_NAMES = ['ADI','BACK','DEB','LYM','MUC','MUS', 'NORM', 'STR', 'TUM']
    elif root_dir.find('Retino') >=0:
        CLASS_NAMES = ['0', '1', '2','3','4']
    elif root_dir.find('ColonPath') >=0:
        CLASS_NAMES = ['0', '1']
    elif root_dir.find('NeoJaundice') >=0:
        CLASS_NAMES = ['0', '1']
    elif root_dir.find('MNIST') >=0:
        CLASS_NAMES = ['0', '1', '2','3','4', '5', '6','7','8','9']
    elif root_dir.find('CIFAR10') >=0:
        CLASS_NAMES = ['0', '1', '2','3','4', '5', '6','7','8','9']
    elif root_dir.find('Dog_Emotions') >=0:
        CLASS_NAMES = ['angry', 'happy','relaxed','sad']
    elif root_dir.find('COVID') >= 0:
        CLASS_NAMES = ['COVID-19', 'Non-COVID', 'Normal']
    elif root_dir.find('CRC') >= 0:
        CLASS_NAMES = ['ADI','BACK','DEB','LYM','MUC','MUS', 'NORM', 'STR', 'TUM']
    elif root_dir.find('DR') >= 0:
        CLASS_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    elif root_dir.find('Tumor') >= 0:
        CLASS_NAMES = ['glioma-tumor', 'meningioma-tumor', 'no-tumor', 'pituitary-tumor']
    elif root_dir.find('Breast') >= 0:
        CLASS_NAMES = ['benign', 'malignant', 'normal']
    elif root_dir.find('Chest_Canser') >= 0:
        CLASS_NAMES = ['adenocarcinoma', 'large-cell-carcinoma', 'normal', 'squamous-cell-carcinoma']
    elif root_dir.find('Skin') >= 0:
        CLASS_NAMES = ['benign', 'malignant']
    return CLASS_NAMES

#CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
class CenDataset(Dataset):
    def __init__(self, root_dirs, N_CLASSES, resolution, beta=None, N_GDATA=None, transform=None):
        super(CenDataset, self).__init__()
        
        self.root_dirs = root_dirs
        self.imgs = []
        self.labels = []

        
        for root_dir in root_dirs:
            CLASS_NAMES = obtainClassNames(root_dir)
            for i in range (0, N_CLASSES):
                imgPath = os.path.join(root_dir, str(i) + '_' + CLASS_NAMES[i])
                if not os.path.exists(imgPath):
                    imgPath = os.path.join(root_dir, str(i) + '_' + str(i))
                if not os.path.exists(imgPath):
                    imgPath = os.path.join(root_dir, CLASS_NAMES[i])
                inames = os.listdir(imgPath)

                if N_GDATA is not None and imgPath.find('diffusion') > 0:
                    train_nums = N_GDATA if N_GDATA < len(inames) else len(inames)
                    print('train_nums:', train_nums)
                else:
                    train_nums = len(inames)
                
                for idx in range (train_nums):    #len(inames)
                    fileName = os.path.join(imgPath, inames[idx])
                    if beta is not None:
                        if 'DDPM' in root_dir:
                            label = np.zeros(N_CLASSES)
                            for id in range(N_CLASSES):
                                label[id] = (1.0 - beta) / (N_CLASSES - 1) if id != i else 1.0*beta
                        else:
                            label = np.zeros(N_CLASSES)
                            label[i] = 1.0
                    else:
                        label = i
                    #numpy_img = Image.open(path)
                    img = Image.open(fileName).convert('RGB')
                    img = img.resize((resolution, resolution))
                    numpy_img = np.array(img)
                    #print(numpy_img[100:110,100:110,1])
                    image = normalize(numpy_img)
                    image = np.transpose(image, [2, 0, 1])
            
                    self.imgs.append(image)
                    self.labels.append(label)
        #print(self.imgNames)
        
        self.imgs = np.array(self.imgs)
        self.weights = np.ones(N_CLASSES)

        self.transform = transform
        print(self.imgs.shape)
        print('Total # labels:{}'.format(len(self.labels)))

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.imgs[idx]
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.labels)
    
    def getWeights(self):
        return self.weights

class Dataset(Dataset):
    def __init__(self, root_dir, N_CLASSES, resolution, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()
        
        self.root_dir = root_dir
        self.imgs = []
        self.labels = []

        CLASS_NAMES = obtainClassNames(root_dir)
        for i in range (0, N_CLASSES):
            imgPath = os.path.join(root_dir, str(i) + '_' + CLASS_NAMES[i])
            inames = os.listdir(imgPath)

            for idx in range (len(inames)):    #len(inames)
                fileName = os.path.join(imgPath, inames[idx])
                label = i
                
                #numpy_img = Image.open(path)
                img = Image.open(fileName).convert('RGB')
                img = img.resize((resolution, resolution))
                numpy_img = np.array(img)
                #print(numpy_img[100:110,100:110,1])
                image = normalize(numpy_img)
                image = np.transpose(image, [2, 0, 1])
        
                self.imgs.append(image)
                self.labels.append(label)
        #print(self.imgNames)
        
        self.imgs = np.array(self.imgs)
        self.weights = np.ones(N_CLASSES)

        self.transform = transform
        print(self.imgs.shape)
        print('Total # labels:{}'.format(len(self.labels)))

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.imgs[idx]
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.labels)
    
    def getWeights(self):
        return self.weights


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class BaseDataset(Dataset):
    """Base dataset iterator"""

    def __init__(self, x, y):
        self.imgs = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]