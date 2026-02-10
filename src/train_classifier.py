import argparse
import sys
import logging
import coloredlogs
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


from utils.loaders import DatasetEmbeddings
from models.embeddings_classification import MLP, LSTM, Transformer

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    parser.add_argument('--model', default="MLP", type=str, help='Model used for training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.005, type=float, help='Initial learning rate')
    parser.add_argument('--batch_size', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    help='Please specify path to the ImageNet training data.')
    parser.add_argument("--logfile", default="training_log", type=str, help="Specify the name of the log file.")
    
    return parser

def setup_logger(name, logfile=None):

    logger_instance = logging.getLogger(name)
    i_handler = logging.FileHandler(logfile)
    i_handler.setLevel(logging.INFO)
    logger_instance.addHandler(i_handler)
    coloredlogs.install(
        level='DEBUG', logger=logger_instance,
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s')
    return logger_instance


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    print("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def evaluate(model, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    accuracy = (all_predictions == all_targets).mean()
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    roc_auc = roc_auc_score(all_targets, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'false_positives': fp,
        'false_negatives': fn,
        'specificity': specificity
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()

    logger = setup_logger("LOG", args.logfile)
    sys.excepthook = handle_exception

    logger.info("_______TRAINING LOG FILE______")

    ############# Device Setup #############
    if torch.backends.mps.is_available():
            DEVICE = 'mps'
            logger.info("------ USING APPLE SILICON GPU ------")
    elif torch.cuda.is_available():
            DEVICE = 'cuda'
            logger.info("------ USING CUDA GPU ------")
    else:
            DEVICE = 'cpu'
            logger.info("------ USING CPU ------")


    ############# Data Loading #############    
    if args.data_path is None:
        raise ValueError("Please specify path to training data.")
    logger.info("Loading data from: {}".format(args.data_path))
    # Load the dataset
    train_dataset = DatasetEmbeddings(args.data_path, split_type='train')
    test_dataset = DatasetEmbeddings(args.data_path, split_type='test')
    logger.info("Dataset loaded successfully.")
    logger.info("Number of samples for training: {}".format(len(train_dataset)))
    logger.info("Number of samples for testing: {}".format(len(test_dataset)))
    logger.info("Number of classes: {}".format(len(set(train_dataset.labels))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ############# Model Setup #############
    embedding_dim = train_dataset.__getitem__(0)[0].shape[0]
    num_class = 2
    if args.model == "MLP": 
        model = MLP(emb_dim=embedding_dim, num_class=num_class)
    elif args.model == "LSTM":
        model = LSTM(emb_dim=embedding_dim, num_class=num_class)
    elif args.model == "Transformer":
        model = Transformer(emb_dim=embedding_dim, num_classes=num_class)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    model.to(DEVICE)
    logger.info("...Model loaded successfully.")
    logger.info("Model architecture: {}".format(model))

    ############# Training Setup #############
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max è il numero di passi fino al minimo

    ############# Training Loop #############
    model.train()
    epoch_loss = [0.0, 0]

    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(train_loader):
            model.train()
            # Training loop
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss[0] += loss.item()
            epoch_loss[1] += data.size(0)

            if (i + 1) % (len(train_loader) // 5) == 0:
                logger.info("[{}/{}]".format(i + 1, len(train_loader)))

        scheduler.step()
        logger.info(f'[EPOCH {epoch+1}] Avg. Loss: {epoch_loss[0] / epoch_loss[1]}')

    # validation loop
        #save checkpoint in a file
        if (epoch+1) % 10 == 0:
            train_metrics = evaluate(model, train_loader, DEVICE)
            val_metrics = evaluate(model, test_loader, DEVICE)
            logger.info(f'[EPOCH {epoch+1}] Train Metrics: {train_metrics}')
            logger.info(f'[EPOCH {epoch+1}] Val Metrics: {val_metrics}')
            torch.save(model.state_dict(), f'./src/ckpts/{args.model}/model_epoch_{epoch+1}.pth')
            logger.info(f'Current LR: {scheduler.get_last_lr()}')



    


