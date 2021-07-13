import time
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().cpu().numpy().mean()
    return acc


def save_model(model, path):
    torch.save(model.state_dict(), path)


def timeLapse(origin, actual):
    return actual - origin


def splitDataset(dataset: Dataset, train=90, test=5, val=5):
    len = dataset.__len__()
    propsum = train + test + val
    longtrain = int(train * len / propsum)
    longtest = int(test * len / propsum)
    longval = int(len - longtrain - longtest)
    return torch.utils.data.random_split(dataset, [longtrain, longtest, longval])


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def tensor_to_image(tensor):
    return functional.to_pil_image(tensor)


def resize(config, image):
    if config["imageResize"]:
        image = cv2.resize(
            image, (config["imageResizeWidth"], config["imageResizeHeight"])
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def loadVideoFrames(config, paths, forcedShowImage=False):
    w = config["imageInputWidth"]
    h = config["imageInputHeight"]
    if config["imageResize"]:
        w = config["imageResizeWidth"]
        h = config["imageResizeHeight"]

    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    minFrames = config["minFrames"]
    xv = torch.FloatTensor(0, minFrames, config["imageChannels"], h, w)
    nullImage = torch.zeros(
        [1, xv.size()[2], xv.size()[3], xv.size()[4]], dtype=torch.float32
    )
    for xi in paths:
        vidcap = cv2.VideoCapture(xi)
        success, imageR = vidcap.read()
        frames = torch.FloatTensor(0, config["imageChannels"], h, w)
        frameNumber = 0
        i = 0
        while success and frameNumber < minFrames:
            if i % config["frameRate"] == 0:
                image = functional.to_tensor(
                    showImage(config, resize(config, imageR), forcedShowImage)
                )
                image = addMomentToTensor(image)
                image = transform(image)
                image = image.unsqueeze(0)
                frames = torch.cat((frames, image))
                frameNumber += 1
            i += 1
            success, imageR = vidcap.read()
        framesToAdd = minFrames - frameNumber
        while framesToAdd > 0:
            frames = torch.cat((frames, nullImage))
            framesToAdd -= 1

        frames = frames.unsqueeze(0)
        xv = torch.cat((xv, frames))
    return xv


def explode(y, elements):
    """
    With a tensor y returns the same tensor exploded at the dimension with elements size. Image that you have a tensor 50,0 and you want to change to 50,20,0 on every value of the new 20 dimension is the same of the previous 0
    """
    result = torch.zeros(0, elements, dtype=torch.int64)
    for ty in y:
        preresult = torch.zeros(0, dtype=torch.int64)
        for i in range(elements):
            preresult = torch.cat((preresult, ty.unsqueeze(0)))
        result = torch.cat((result, preresult.unsqueeze(0)))
    return result


def addMomentToTensor(tensor, moment=time.time()):
    """
    Adds info from now on a tensor as a last tensor position
    """
    if tensor.size()[0] < 3:
        tensor[0, 0, 0, 0] = moment
        tensor[0, 1, 0, 0] = moment
        tensor[0, 2, 0, 0] = moment
    else:
        tensor[0, 0, 0] = moment
        tensor[1, 0, 0] = moment
        tensor[2, 0, 0] = moment
    return tensor


def sendToCuda(config, x, y, device):
    return x, y.to(device)


def removeFromCuda(config, x, y, z):
    return x, y.cpu(), z.cpu()


class splitVideo:
    def __init__(self, config, x, y):
        self.config = config
        self.actual = 0
        self.x = x
        self.y = y
        self.loadAsImages = config["loadAsImages"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.loadAsImages:
            if self.actual > 0:
                raise StopIteration
            else:
                self.current += 1
                return self.x, self.y
        else:
            if self.actual == 0:
                self.vidcap = cv2.VideoCapture(self.x[self.actual])
                self.actual += 1
                success, image = self.vidcap.read()
                return (
                    functional.to_tensor(showImage(resize(self.config, image))),
                    self.y,
                )
            else:
                success, image = self.vidcap.read()
                if not success:
                    self.actual += 1
                    if self.actual >= len(self.x):
                        raise StopIteration
                    else:
                        self.vidcap = cv2.VideoCapture(self.x[self.actual])
                        success, image = self.vidcap.read()
                        return (
                            functional.to_tensor(showImage(resize(self.config, image))),
                            self.y,
                        )

                else:
                    return (
                        functional.to_tensor(showImage(resize(self.config, image))),
                        self.y,
                    )


def showImage(config, image, forcedShowImage=False):
    if config["showImages"] or forcedShowImage:
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)
    return image


def log(logFile, message):
    logFile.write(f"{message}\n")
    logFile.flush()
    print(message)


def saveLog(config, sufix, loss, acc):
    logFile = open(f"{config['root']}.{sufix}", "a")
    logFile.write(f"[{loss},{acc}],\n")
    logFile.close()


def getNow():
    return datetime.now().strftime("%Y-%d-%m %H:%M:%S")


def optimizerTo(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
