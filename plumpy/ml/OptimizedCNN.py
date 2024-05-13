'''
Currently: outer CV for accuracy is part of the optimized SVM class:

- all data are fed to SVM
- it is split based on groups of classes: a unique set of 12 in each split => 10 splits: 9 repetitions in train, 1 in test
- per each split
    - a parallel thread is set up for speedup
    - an Optuna study is set up using only train (9 repetitions)
    - ShuffleSplit: 80% train_, 20% test_ is used in Optuna trials, 150 trials in total
    - best param is used on the test set data
- results are aggregated over 10 splits

Better to rearrange?
    - have a separate ParallelCV class that makes up the 10 splits and runs a function per split: parallelCV.run()
    - run optimizedSVM.train() per split
    - aggregate results over splits

'''

import gc
import typing
import time
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from riverfern.utils.general import flatten_4d_data
from subprocess import check_output, CalledProcessError
from sklearn.metrics import accuracy_score as cpu_acc
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit
from sklearn.svm import SVC as sksvc

#from multiprocessing import Process, Value

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NNClassifier():
    def __init__(self, model):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.model = model

    def train(self, trainloader):
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

class OptimizedCNN:
    def __init__(self, id: str, cBounds: typing.Tuple = (1e-5, 1e5),
                       n_optuna_trials: int = 150,
                       optuna_cv: str = 'group',
                       probability: bool = True,
                       forceCPU: bool = True) -> None:
        self.id = id # subject ID or similar for logs, TODO: make it optional
        self.probability = probability
        self.optuna_cv = optuna_cv
        self.n_optuna_trials = n_optuna_trials
        self._cLBound = 1e-2 if cBounds is None else cBounds[0]
        self._cUBound = 1e2 if cBounds is None else cBounds[1]
        # check whether GPU is available on the current system
        try:
            out = check_output('nvidia-smi')
            if out != "Failed to initialize NVML: Driver/library version mismatch":
                self._force_CPU = False
                print("SVC: NVidia GPU and drivers detected.")
            else:
                self._force_CPU = True
                print("SVC: No NVidia GPU or drivers detected.")
        except (CalledProcessError, FileNotFoundError) as error:
            self._force_CPU = True
            print("SVC: No NVidia GPU or drivers detected.")
        if forceCPU: self._force_CPU = forceCPU
        if self._force_CPU:
            print("SVC: GPU training disabled.")
        else:
            print("SVC: GPU training enabled.")
        self._study_enumerator = range(0)

    def train(self, x, y, groups=None, n_jobs=1):
        '''
        :param x:
        :param y:
        :param groups:
        :param n_jobs:
        :return:
        '''

        # permute x and labels to avoid any sequencing effects
        # use deterministic seed
        p = np.random.RandomState(seed=760).permutation(y.shape[0])
        x = x[p]
        y = y[p]

        # flatten the data: has to happen here for the grids to handle electrode selection
        x = flatten_4d_data(x)

        #storage = JournalStorage(JournalFileStorage("optuna-journal.log"))
        self.optuna_study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=142), # TODO: check seed
                                    #storage=storage,
                                    study_name=self.id + "_OptunaCV",
                                    direction="maximize")

        # optimize using optuna
        n_trials = self.n_optuna_trials

        #splitinds = [t for t in ShuffleSplit(n_trials, test_size=0.2).split(x, y)]
        if self.optuna_cv == 'group':
            splitinds = [t for t in GroupShuffleSplit(n_trials, test_size=.1, random_state=800).split(x, y, groups=groups)]
        elif self.optuna_cv == 'stratified':
            splitinds = [t for t in StratifiedShuffleSplit(n_trials, test_size=.1, random_state=800).split(x, y, groups=groups)]
        elif self.optuna_cv == 'shuffled':
            splitinds = [t for t in ShuffleSplit(n_trials, test_size=0.1, random_state=800).split(x, y)]
        else:
            raise NotImplementedError
        # np.sort(events.labels[ind_train][splitinds[0][1]])

        self.optuna_study.optimize(lambda trial: self._objective(trial, x, y, splitinds, False), n_trials=n_trials,
                       n_jobs=n_jobs)
        self.optuna_result = self.optuna_study.trials_dataframe()
        self.optuna_best = self.optuna_study.best_params

        # train final classifier on GPU if the set is large and GPU is available
        # I have determined by trial and error that the point at which the GPU overhead becomes too large is around 150
        if (x.shape[1] > n_trials * 8 and not self._force_CPU): # self._window.Length
            try:
                self._trainGPUSVM(x, y, self.optuna_study.best_params["C"])
            except RuntimeError:  # GPU out of memory
                print("SVC: GPU out of memory. Fall back to CPU training.")
                self._trainCPUSVM(x, y, self.optuna_study.best_params["C"])
        else:
            self._trainCPUSVM(x, y, self.optuna_study.best_params["C"])

        # clear GPU memory
        gc.collect()


    def _objective(self, trial: optuna.Trial, X, y, splitinds: typing.Collection[tuple],
                   log_times: bool = False) -> float:
        """Objective function for the optuna study. Uses accuracy of the classifier as quality measure and the suggests a float value for C.
        Using a predefined set of splitindices is much faster than using a train test split everytime.

        Args:
            - trial (optuna.Trial): Trial object of the optuna study
            - X (iterable): Dataframe for GPU computing containing the feature set.
            - y (iterable): Dataframe for GPU computing containing the labels. Must be encoded to only contain numeric values.
            - splitinds (typing.Union[typing.Collection[tuple],None], optional): Predefined indices for train/test splitting. Recommended to be provided to save significant amounts of time. If not provided fall back to cuML train_test_split function. Defaults to None.

        Returns:
            - float: Accuracy of the classifier
        """
        if log_times: tic = float(time.perf_counter())
        # check if it is possible to define a step in a more suitable manner, delted for now
        c = trial.suggest_float("C", self._cLBound, self._cUBound)
        # create new instance of classifier. Compared to older versions of this code a new instance has to be created to avoid
        # memory overflow because calling the fit method twice will not override the old classifier in cuML
        #clf = sksvc(kernel=self.kernel, C=c, decision_function_shape="ovo")
        clf = CNN()

        train, test = splitinds[trial.number]
        if log_times:
            toc = float(time.perf_counter())
            endbp = float(toc - tic)
            print("job " + str(trial.number) + ": end bp: " + str(endbp))

        clf.fit(X[train, :], y[train])
        if log_times:
            tic = float(time.perf_counter())
            endclf = float(tic - toc)
            print("job " + str(trial.number) + ": end clf: " + str(endclf))

        y_pred = clf.predict(X[test])
        acc = cpu_acc(y[test], y_pred)
        if log_times:
            toc = float(time.perf_counter())
            print("job " + str(trial.number) + ": end acc: " + str(toc - tic))

        # gc.collect() # necessary to be able to use cupy/numba arrays. If missing will cause a memory leak
        return acc

    def _trainCPUSVM(self, dfx, dfy, c: float):
        self.clf = sksvc(kernel=self.kernel, C=c, decision_function_shape="ovo", probability=self.probability)
        self.clf.fit(dfx, dfy)

    # TODO: combine with trainCPU, can just branch, add funs _make_classifier and _get_coef
    def _trainGPUSVM(self, dfx, dfy, c: float):
        # import cupy
        from cuml.internals.memory_utils import set_global_output_type
        from cuml.common.device_selection import set_global_device_type
        from cuml.svm import SVC as cusvc
        # monitor = Monitor(10)
        set_global_output_type('numpy')  # set the output type of the cuml library
        set_global_device_type('gpu')
        self.clf = cusvc(kernel="linear", C=c)
        # self.clf.fit(dfx, self._le.transform(dfy))
        # clf.fit(cupy.array(dfx), cupy.array(self._le.transform(dfy)))
        # monitor.stop()
        print('train')
        #_predLabels = list(self._le.inverse_transform(clf.predict(x[test, :])))
        #_estimatorCoefs = self.extractCuMlSvcCoefs(clf)
        #return (_predLabels, _estimatorCoefs)

    def test(self, x_test, y_test):
        x_test = flatten_4d_data(x_test)

        _predLabels = list(self.clf.predict(x_test))
        try:
            _estimatorCoefs = self.clf.coef_ # only for linear kernel
        except:
            _estimatorCoefs = None
        _score = self.clf.score(x_test, y_test)
        return _score, _predLabels, _estimatorCoefs

    def get_proba(self, x_test):
        if self.probability == True:
            x_test = flatten_4d_data(x_test)
            return self.clf.predict_proba(x_test)
        else:
            NotImplementedError

    def extractCuMlSvcCoefs(clf):
        """Replicates the behaviour of sklearn when calling .coef_ property of a multiclass estimator object from the CUML library which
        fails if called natively

        Args:
            - clf (cuML SVC estimator): The CUML SVC model trained on the data which contains the multiclass estimator

        Returns:
            - np.ndarray: The weight matrix as it would be returned by SKLearn with dimensions n_estimators x n_features (usually 66 x (n_channels*n_timepoints))
        """
        return np.vstack([c.coef_ for c in clf.multiclass_svc.multiclass_estimator.estimators_])
