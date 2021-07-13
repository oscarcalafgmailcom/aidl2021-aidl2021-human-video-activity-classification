from torch.utils.data import DataLoader
from NTUDataset import NTUDataset
import utils as utils

if __name__ == "__main__":

    # Hyperparameters
    config = {
        "lr": 0.0001,
        "batch_size": 20,
        "epochs": 25,
        "h1": 16,
        "h2": 32,
        "h3": 64,
        "h4": 1024,
        "h5": 512,  # not in use
        "features": 60,
        "imageChannels": 3,
        "imageInputWidth": 1920,
        "imageInputHeight": 1080,
        "imageResize": True,
        "showImages": True,
        "imageResizeWidth": 240,  # 240
        "imageResizeHeight": 136,  # 136
        "frameRate": 5,
        "minFrames": 20,
        "model": "3DCNN",  # 'LSTMONLYEND' 'VGG16LSTModel' OR 'CNN' OR 'CNNLSTM' OR '3DCNN'
        "videoPatterns": [
            # "/data/nturgb+d_rgb/*.avi",
            "/Volumes/Disc Extern/nturgb+d_rgb/S002C002P014R001A001_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*02_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*03_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*04_rgb.avi",
        ],
    }
    dataset = NTUDataset(
        config["videoPatterns"],
        config,
    )
    train_dataset, val_dataset, test_dataset = utils.splitDataset(dataset, 100, 0, 0)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    for x, y in train_loader:
        utils.loadVideoFrames(config, x, forcedShowImage=True)
