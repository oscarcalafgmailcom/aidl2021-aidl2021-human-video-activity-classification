from models.CNNModel import CNNModel
from models.VGG16LSTMOnlyEndModel import VGG16LSTMOnlyEndModel
from models.CNNLSTMModel import CNNLSTMModel
from models.CNNLSTMSumModel import CNNLSTMSumModel
from models.n3DCNNModel import n3DCNNModel
from models.CNNLSTM import CNNLSTM
from models.RESNET101LSTMModel import RESNET101LSTMModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from NTUDataset import NTUDataset
import utils as utils
import numpy as np
import gc
from datetime import datetime
import time
import os.path
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config, model, train_loader, criterion, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        x, y = utils.sendToCuda(config, x, y, device)
        optimizer.zero_grad()
        model.hidden = None
        # We send y to the model to "arrange" y when is necessary for the model
        y_hat, y = model(x, y)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        utils.removeFromCuda(config, x, y, y_hat)
        acc = utils.accuracy(y, y_hat)
        losses.append(loss.item())
        accs.append(acc.item())
        print(f"              TRAIN  loss:{loss.item()} acc:{acc.item()}")
        utils.saveLog(config, "train", loss.item(), acc.item())
        del x, y, y_hat, loss, acc
        gc.collect()

    return np.mean(losses), np.mean(accs)


def test(config, model, dataloader: DataLoader, criterion):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = utils.sendToCuda(config, x, y, device)
            y_hat, y = model(x, y)
            loss = criterion(y_hat, y)
            utils.removeFromCuda(config, x, y, y_hat)
            acc = utils.accuracy(y, y_hat)
            losses.append(loss.item())
            accs.append(acc.item())
            print(f"              TEST  loss:{loss.item()} acc:{acc.item()}")
            utils.saveLog(config, "test", loss.item(), acc.item())
            del x, y, y_hat, loss, acc
            gc.collect()

    return np.mean(losses), np.mean(accs)


if __name__ == "__main__":

    # Hyperparameters
    config = {
        "lr": 0.001,
        "weight_decay": 0.01,
        "dropOut": 0.5,
        "batch_size": 50,
        "epochs": 100,
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
        "showImages": False,
        "imageResizeWidth": 240,  # 480,  # 240
        "imageResizeHeight": 136,  # 272,  # 136
        "frameRate": 15,
        "minFrames": 16,
        "model": "VGG16LSTMOnlyEndModel",  # 'CNNLSTMModel' 'CNNLSTMSumModel' 'VGG16LSTMOnlyEndModel' 'VGG16LSTMModel' OR 'CNN' OR 'CNNLSTM' OR '3DCNN'
        "videoPatterns": [
            "/data/nturgb+d_rgb/*.avi",
            # "/data/nturgb+d_rgb/*01_rgb.avi",
            # "/data/nturgb+d_rgb/*02_rgb.avi",
            # "/data/nturgb+d_rgb/*03_rgb.avi",
            # "/data/nturgb+d_rgb/*04_rgb.avi",
            # "/data/nturgb+d_rgb/*05_rgb.avi",
            # "/data/nturgb+d_rgb/*06_rgb.avi",
            # "/data/nturgb+d_rgb/*07_rgb.avi",
            # "/data/nturgb+d_rgb/*08_rgb.avi",
            # "/data/nturgb+d_rgb/*09_rgb.avi",
            # "/data/nturgb+d_rgb/*10_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*01_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*02_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*03_rgb.avi",
            # "/Volumes/Disc Extern/nturgb+d_rgb/*04_rgb.avi",
        ],
    }

    startTime = time.time()

    initialEpoch = 0
    trainLosses, trainAccuracies = [], []
    testLosses, testAccuracies = [], []

    superRoot = f"{config['model']}"
    _continueRunning = False
    root = f"results/{superRoot}"
    checkPointFileName = f"{root}.pt"
    if os.path.isfile(checkPointFileName):
        _continueRunning = True

    root = f"results/{superRoot}"
    if _continueRunning:
        logFile = open(f"{root}.log", "a")
        checkpoint = torch.load(checkPointFileName)
        trainAccuracies, trainLosses = (
            checkpoint["trainAccuracies"],
            checkpoint["trainLosses"],
        )
        testAccuracies, testLosses = (
            checkpoint["testAccuracies"],
            checkpoint["testLosses"],
        )
        initialEpoch = checkpoint["initialEpoch"]
    else:
        logFile = open(f"{root}.log", "w+")

    config["root"] = root
    config["superRoot"] = superRoot
    # Load the dataset
    # Aquí tenim els path als videos i el label de la classificació
    # /Volumes/Disc Extern/nturgb+d_rgb/*1_rgb.avi
    # ../nturgb+d_rgb/*1_rgb.avi
    dataset = NTUDataset(
        config["videoPatterns"],
        config,
    )
    train_dataset, val_dataset, test_dataset = utils.splitDataset(dataset, 80, 10, 10)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    model = CNNModel(config)
    if config["model"] == "VGG16LSTMOnlyEndModel":
        model = VGG16LSTMOnlyEndModel(config, device=device)
    if config["model"] == "CNNLSTM":
        model = CNNLSTM(config, device=device)
    if config["model"] == "3DCNN":
        model = n3DCNNModel(config, device=device)
    if config["model"] == "CNNLSTMModel":
        model = CNNLSTMModel(config, device=device)
    if config["model"] == "CNNLSTMSumModel":
        model = CNNLSTMSumModel(config, device=device)
    if config["model"] == "RESNET101LSTMModel":
        model = RESNET101LSTMModel(config, device=device)

    if not _continueRunning:
        utils.log(logFile, f"Config: {config}\n")
        utils.log(logFile, f"Model: {model}")
        # summaryText = summary(model,input_size=(config['minFrames'],3,config['imageResizeHeight'],config['imageResizeWidth']),batch_size=config['batch_size'])
        summaryText = ""
        utils.log(logFile, summaryText)
        utils.log(logFile, f"Parameters: {sum(p.numel() for p in model.parameters())}")
        utils.log(
            logFile,
            f"Parameters trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
        )

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    if _continueRunning:
        model.load_state_dict(checkpoint["modelStateDict"])
        optimizer.load_state_dict(checkpoint["optimizerStateDict"])
        utils.optimizerTo(optimizer, device)

    criterion = nn.CrossEntropyLoss()
    utils.log(
        logFile,
        f"Videos: {str(len(dataset.files))} train:{len(train_dataset)} validation:{len(val_dataset)} test:{len(test_dataset)}",
    )

    for epoch in range(initialEpoch, config["epochs"]):
        model = model.to(device)

        lapseTime = time.time()
        loss, acc = train(config, model, train_loader, criterion, optimizer)
        losst, acct = test(config, model, val_loader, criterion)
        lapseTime1 = time.time()
        utils.log(
            logFile,
            f"Epoch {epoch}/{config['epochs']} [{utils.timeLapse(lapseTime, lapseTime1)}] lossTrain={loss:.6f} accTrain={acc:.6f} lossVal={losst:.6f} accVal={acct:.6f}",
        )
        # CheckPoint save at every epoch
        utils.saveLog(config, "epochTrain", loss, acc)
        utils.saveLog(config, "epochTest", losst, acct)
        trainLosses.append(loss), trainAccuracies.append(acc)
        testLosses.append(losst), testAccuracies.append(acct)
        checkpoint = {
            "config": config,
            "modelStateDict": model.cpu().state_dict(),
            "optimizerStateDict": optimizer.state_dict(),
            "trainLosses": trainLosses,
            "trainAccuracies": trainAccuracies,
            "testLosses": testLosses,
            "testAccuracies": testAccuracies,
            "initialEpoch": epoch + 1,
        }
        torch.save(checkpoint, f"{root}.pt")

    lapseTime = time.time()
    loss, acc = test(config, model, test_loader, criterion)
    lapseTime1 = time.time()
    utils.log(
        logFile,
        f"[{utils.timeLapse(lapseTime, lapseTime1)}] TEST LOSS={loss:.6f} TEST ACCURACY={acc:.6f}",
    )

    # Now save the artifacts of the training
    # We can save everything we will need later in the checkpoint.
    checkpoint = {
        "modelStateDict": model.cpu().state_dict(),
        "optimizerStateDict": optimizer.state_dict(),
        "trainLosses": trainLosses,
        "trainAccuracies": trainAccuracies,
        "testLosses": testLosses,
        "testAccuracies": testAccuracies,
        "initialEpoch": epoch,
        "TRAINACCURACY": acc,
        "TRAINLOSS": loss,
    }
    torch.save(checkpoint, f"{root}.pt")

    utils.log(
        logFile,
        f"Total time {utils.timeLapse(startTime, lapseTime)} {startTime} {lapseTime}",
    )
    utils.log(logFile, f"END")
    logFile.close()
