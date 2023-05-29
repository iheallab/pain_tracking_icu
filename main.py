import datetime
from time import time
from sklearn.utils import class_weight

import math
import random
import torch
import numpy as np
import json
import logging
from os.path import join

from util.utils import get_metrics, print_metrics, start_timer, end_timer_and_print, validation
from util.utils import load_data, get_loaders, get_class_distribution
from util.utils import set_model, split_folders, get_labels, plot_cf
from util import utils


def train(model, config, device, train_loader, train_labels, use_cuda, n_classes, val_loader=None):

    start_timer()
    # Get training details
    n_freq_print = config.get("n_freq_print")
    n_epochs = config.get("n_epochs")
    # Load the dataset
    # Set to train mode
    # Load the checkpoint if needed
    logging.info("Initializing from scratch")


    # Set the loss
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss = torch.nn.NLLLoss(weight=class_weights)

    # Set the optimizer and scheduler
    optim = torch.optim.Adam(model.parameters(),
                             lr=config.get('lr'),
                             eps=config.get('eps'),
                             weight_decay=config.get('weight_decay'))
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                step_size=config.get('lr_scheduler_step_size'),
                                                gamma=config.get('lr_scheduler_gamma'))

    # Train
    checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
    logging.info("Start training")
    best_loss = 1000000
    losses = []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(n_epochs):
        model.train(True)
        start = time()
        loss_vals = []
        for minibatch, label in train_loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                minibatch["acc"] = minibatch["acc"].to(device)
                minibatch["clin"] = minibatch["clin"].to(device)
                label = label.to(device)
                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Collect for recoding and plotting
                batch_loss = criterion.item()
                loss_vals.append(batch_loss)

            # Back prop
            scaler.scale(criterion).backward()
            scaler.step(optim)
            # Updates the scale for next iteration.
            scaler.update()
            optim.zero_grad()
        losses.append(np.mean(loss_vals))
        # Scheduler update
        scheduler.step()
        if epoch % n_freq_print == 0:
            # Plot the loss function
            loss_fig_path = checkpoint_prefix + "_loss_fig.png"
            utils.plot_loss_func(np.arange(0, epoch + 1), losses, loss_fig_path)

        # Record loss on train set
        model.train(False)
        if val_loader is not None:
            current_loss, current_metric = validation(model, val_loader, device, n_classes, use_cuda=use_cuda)
        end = time()
        if val_loader is not None:
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(n_epochs))).format(epoch + 1)
            epoch_fin = 'train: {:.6f}, val: {:.6f}, F1: {:.4f} [{}]'.format(np.mean(loss_vals), current_loss,
                                                                             current_metric['f1-score_macro'],
                                                                             str(datetime.datetime.timedelta(
                                                                                 seconds=(end - start))))
            logging.info(epoch_desc + epoch_fin)
            if best_loss > current_loss:
                best_loss = current_loss
                torch.save(model.state_dict(), checkpoint_prefix + "_best.pth")
                logging.info("Best model saved")
        else:
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(n_epochs))).format(epoch + 1)
            epoch_fin = 'train: {:.6f} [{}]'.format(np.mean(loss_vals), str(datetime.timedelta(seconds=(end - start))))
            logging.info(epoch_desc + epoch_fin)
            # Save checkpoint
            if best_loss > np.mean(loss_vals):
                best_loss = np.mean(loss_vals)
                torch.save(model.state_dict(), checkpoint_prefix + "_best.pth")
                logging.info("Best model saved")

    end_timer_and_print(f"Training session ({n_epochs}epochs)")

    logging.info('Training completed')
    torch.save(model.state_dict(), checkpoint_prefix + '_final.pth')


def test(config, device, device_id, test_loader, folder_idx, n_classes):
    checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())

    model = set_model(config, device, n_classes)
    if config.get("checkpoint_path") != "None":
        model.load_state_dict(torch.load(checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(checkpoint_path))
    else:
        # Load the best model
        if config.get("best_model"):
            model.load_state_dict(torch.load(checkpoint_prefix + "_best.pth", map_location=device_id))
        else:
            model.load_state_dict(torch.load(checkpoint_prefix + "_final.pth", map_location=device_id))
    # Set to eval mode
    model.eval()

    logging.info("Start testing")
    predicted = []
    ground_truth = []

    with torch.no_grad():
        for minibatch, label in test_loader:
            # Forward pass
            minibatch["acc"] = minibatch["acc"].to(device)
            minibatch["clin"] = minibatch["clin"].to(device)
            label = label.to(device)
            res = model(minibatch)

            # Evaluate and append
            pred_label = torch.argmax(res, dim=1)
            predicted.extend(pred_label.cpu().numpy())
            ground_truth.extend(label.cpu().numpy())

    def plot_metrics(ground_truth, predicted, n_classes):
        metrics = get_metrics(ground_truth, predicted, n_classes)
        for k, v in metrics.items():
            if "confusion" in k:
                logging.info('Fold {} {}:\n{}\n'.format(folder_idx, k.capitalize(), v))
                plot_cf(v)

            else:
                logging.info('Fold {} {}: {}\n'.format(folder_idx, k.capitalize(), v))
        return metrics
    metrics = plot_metrics(ground_truth, predicted, n_classes)

    return metrics


if __name__ == "__main__":
    # Read configuration
    with open('config.json', "r") as read_file:
        config = json.load(read_file)

    # Set the seeds and the device
    torch_seed = 42
    numpy_seed = 42
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    random.seed(42)

    # Set the dataset and data loader
    logging.info("Start train data preparation")
    # read data and get dataset and dataloader
    # split the data into train, val and test sets
    X, y, y_target, y_col_names = load_data(config.get("data_path"), clin_variable_target="pain_score")

    col_idx_target = y_col_names.index("pain_score")
    col_idx_prevpain = y_col_names.index('pain_score_prev')

    y_target = np.round(np.array(y[:, col_idx_target], dtype=float))
    prev_pain = np.round(np.array(y[:, col_idx_prevpain], dtype=float))

    experiment = config.get("experiment")

    X, y, yy_t, prev_pain_t, num_folders, n_classes = get_labels(X, y, y_target, prev_pain, experiment)

    folders = split_folders(y, experiment)
    print(np.unique(prev_pain_t, return_counts=True))
    print(np.unique(yy_t, return_counts=True))

    sample_start = X.shape[1] - config.get("sample_size")
    checkpoint_path = config.get("checkpoint_path")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_id = config.get('device_id')
    else:
        device_id = 'cpu'

    device = torch.device(device_id)

    utils.init_logger()
    # Record execution details
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    model = set_model(config, device, n_classes)

    cum_acc, cum_recall, cum_precision, cum_auc, cum_f1 = [], [], [], [], []
    cum_recall_macro, cum_precision_macro, cum_f1_macro = [], [], []

    for folder_idx in range(num_folders):
        clin_data = np.expand_dims(prev_pain, axis=1)

        test_idx = folders[folder_idx][0]
        train_idx = folders[folder_idx][1]

        train_acc_data, train_labels, test_acc_data, test_labels = X[train_idx], yy_t[train_idx], X[test_idx], yy_t[test_idx]
        train_clin_data, test_clin_data = clin_data[train_idx], clin_data[test_idx]
        train_data = {"acc": train_acc_data, "clin": train_clin_data}
        test_data = {"acc": test_acc_data, "clin": test_clin_data}

        logging.info(f"Folder {folder_idx + 1}")
        logging.info(f"Train data: {get_class_distribution(np.unique(train_labels, return_counts=True))}")
        logging.info(f"Test data: {get_class_distribution(np.unique(test_labels, return_counts=True))}")

        # get dataloaders
        train_loader, test_loader = get_loaders(config.get("batch_size"), sample_start, train_data, train_labels, test_data, test_labels)

        logging.info("Train data shape: {}".format(train_data["acc"].shape))
        logging.info("Train data shape: {}".format(train_data["clin"].shape))
        if config.get("checkpoint_path") == "None":
            train(model, config, device, train_loader, train_labels, n_classes, use_cuda)
        metrics = test(config, device, device_id, test_loader, folder_idx, n_classes)

        logging.info("Data preparation completed")
        cum_acc.append(metrics['accuracy'])
        cum_f1.append(metrics['f1-score'])
        cum_recall.append(metrics['recall'])
        cum_precision.append(metrics['precision'])
        cum_auc.append(metrics['roc_auc'])
        cum_f1_macro.append(metrics['f1-score_macro'])
        cum_recall_macro.append(metrics['recall_macro'])
        cum_precision_macro.append(metrics['precision_macro'])

    print_metrics(logging, n_classes, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro,
                  cum_precision_macro, cum_f1_macro)
