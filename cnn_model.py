import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1984, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNNet().to(device)

# cost function used to determine best parameters
cost = torch.nn.CrossEntropyLoss()

# used to create optimal parameters
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


class AddIterator:

    def __init__(self, X, y, batch_size=32):
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0
        self.y = y
        self.X = X

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        k_bs = k * bs
        k_bs_1 = (k + 1) * bs

        return self.X[k_bs: k_bs_1], self.y[k_bs: k_bs_1]


def batches(X, y, bs=32):
    for X, y in AddIterator(X, y, bs):
        X = torch.unsqueeze(torch.Tensor(X), 1)
        y = torch.LongTensor(y)
        yield X, y


# Create the validation/test function
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


X_train, y_train = load_data("training_data_fma_small.json")
X_valid, y_valid = load_data("validation_data_fma_small.json")

datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}

label_dict = {7: 0, 12: 1, 6: 2, 13: 3, 5: 4}
y_train = np.asarray([label_dict[i] for i in list(y_train)])
y_valid = np.asarray([label_dict[i] for i in list(y_valid)])

bs = 32
lr = 0.01
n_epochs = 30
patience = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)


def train_model(model, criterion, optimizer, n_epochs, patience, bs):
    history = []
    best_loss = np.inf
    best_weights = None
    no_improvements = 0

    # Training loop
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}

        for phase in ('train', 'val'):
            if phase == 'train':
                training = True
            else:
                training = False

            running_loss = 0

            for batch in batches(*datasets[phase], bs=bs):
                X_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()

                # compute gradients only during 'train' phase
                with torch.set_grad_enabled(training):
                    outputs = model(X_batch)  # feature_names
                    loss = criterion(torch.squeeze(outputs), y_batch)

                    # don't update weights and rates when in 'val' phase
                    if training:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            stats[phase] = epoch_loss

            # early stopping: save weights of the best model so far
            if not training:
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improvements = 0
                else:
                    no_improvements += 1

        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.9f} - val: {val:.9f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break

    return best_weights, history


def evaluate_model(model, y_valid):
    with torch.no_grad():
        pred = model(torch.unsqueeze(y_valid, 1))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.core.series
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    r2_score, mean_absolute_percentage_error, mean_pinball_loss, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import scikitplot as skplt

class EvaluateModel:
    """
    The class ScikitLearnModels include functionalities to train, test and observe test results. It can use
    different types of scikit learn model, as long as parameters send to a random search are expected by the API.
    """

    def __init__(self, X_train, y_train, X_test, y_test, parameters, model_type, experiment_name, regression,
                 multiclass=False,
                 api=None):
        """
        @param X_train: pandas DataFrame
        @param y_train: pandas DataFrame
        @param X_test: pandas DataFrame
        @param y_test: pandas DataFrame
        @param parameters: dictionary of parameters for random search
        @param model_type: any scikit learn model (RandomForestRegressor, GradientBoostingClassifier, etc.)
        """
        self.model_type = model_type
        self.parameters = parameters
        self.columns = X_train.columns
        self.results = None
        self.param_used = None
        self.model = None
        self.models = None
        self.pred_actual = None
        self.scaler = None
        self.alpha = None
        self.api = api
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.regression = regression
        self.multiclass = multiclass
        self.experiment_name = experiment_name

    def get_metrics_multiclass(self, model):

        train_auc = roc_auc_score(y_train_dumm, y_train_pred_prob, multi_class=arg_)
        test_auc = roc_auc_score(y_test_dumm, y_test_pred_prob, multi_class=arg_)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')

        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,

        }

    def get_metrics(self):
        """
        use model to create predictions on test set, then create metrics and store metrics in dict
        @param model: fitted model
        @return: dict with metrics
        """

        train_acc = accuracy_score(y_train, y_train_pred > 0.5)
        test_acc = accuracy_score(y_test, y_test_pred > 0.5)

        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred > 0.5)
        test_f1 = f1_score(y_test, y_test_pred > 0.5)

        train_precision = precision_score(y_train, y_train_pred > 0.5)
        test_precision = precision_score(y_test, y_test_pred > 0.5)

        train_recall = recall_score(y_train, y_train_pred > 0.5)
        test_recall = recall_score(y_test, y_test_pred > 0.5)

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,

        }

    def plot_results(self):
        """
        plot error distribution and prediction vs actual scatter plot
        @param model: fitted scikit learn model
        @param model_index: index of model stored in dictionary
        @return: None
        """
        plt.figure(figsize=(10, 10))
        plt.title(f'Predictions in function of actual values for model of experiment {self.experiment_name}')
        sns.scatterplot(x='Actual', y='Prediction', data=pred_actual)
        plt.savefig(f'Data_Processed/scatterplot_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.title(f'Error distribution for model of experiment {self.experiment_name}')
        sns.histplot(x='Error', data=pred_actual)
        plt.savefig(f'Data_Processed/histplot_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_learning_curves(self):

        plt.figure()
        plt.title('Model learning curves')
        # plot learning curves
        plt.plot(results['validation_0'][metric], label=f'train {metric}')
        plt.plot(results['validation_1'][metric], label=f'test {metric}')
        # show the legend
        plt.legend()
        # show the plot
        plt.savefig(f'Data_Processed/learning_curves_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_probabilities(self, model=None, model_index=None, classes=None):

        if model is None:
            model = self.models[model_index]

        preds = model.predict_proba(self.X_test)

        plt.figure(figsize=(10, 7))
        plt.title('Prediction probability distribution')
        for i, j in zip(list(range(preds.shape[1])), classes):
            plt.hist(preds[:, i], label=j, alpha=0.8)
        plt.legend()
        plt.savefig(f'Data_Processed/pred_proba_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_confusion_matrix(self, classes=None):

        conf_matrix = confusion_matrix(self.y_test, preds)

        if classes:
            conf_matrix = conf_matrix / np.sum(conf_matrix)
            conf_matrix = pd.DataFrame(conf_matrix)
            conf_matrix.columns = classes
            conf_matrix.index = classes

        else:
            conf_matrix = conf_matrix / np.sum(conf_matrix)

        plt.figure(figsize=(10, 7))
        # plt.figure(figsize=(10,7))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(conf_matrix, annot=True,
                    fmt='.2%', annot_kws={"size": 16})  # font size
        plt.savefig(f'Data_Processed/confusion_matrix_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_roc_curve(self, model=None, model_index=None):

        if model is None:
            model = self.models[model_index]

        if self.multiclass:
            probs = model.predict_proba(self.X_test)
            print(probs)

            plt.figure(figsize=(10, 7))
            skplt.metrics.plot_roc(self.y_test, probs)
            plt.savefig(f'Data_Processed/roc_curve_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
            plt.show()


train_model(model, criterion, optimizer, 20, 1, 32)
