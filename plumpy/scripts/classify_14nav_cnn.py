"""

Main script: load data using riverfern -> Epochs acros train. All this data is to be used for optimization (train +
validation). Then, create an Optimizer, pass objective function. The objective function is the trainer that receives
data, parameters for making the model, splits data into train, val, trains the model using params, returns val loss.
The optimizer then optimizer the params. Once the optimization is done, the best set of models is applied to test.
Need:
    - main script (riverfern for loading, setting all up, testing, saving)
    - model class + make_model from params
    - trainer (objective): takes data, params, returns val loss
    - objective (runs trainer)
    - optimizer: sets up optimization study

TODO:
    - augment input data, add shifts around 1 second before and after


"""
import torch
import numpy as np
import tqdm
import pandas as pd
import warnings
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '/home/julia/Documents/Python/RiverFErn')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')

from sklearn.model_selection import train_test_split
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from riverfern.ml.Scaler import Scaler
from riverfern.dataset.Dataset import Dataset, Events
from riverfern.utils.general import sec2ind
from plumpy.utils.io import load_config
from plumpy.utils.general import to_list
from plumpy.ml.models import make_model

warnings.simplefilter(action='ignore', category=FutureWarning)

class Epochs:
    def __init__(self, raw: Dataset, events: Events, tmin: float, tmax: object) -> None:

        self.data_ = OrderedDict({k:[] for k in raw.data.keys()})
        self.data_augmented_ = OrderedDict({k: [] for k in raw.data.keys()})
        self.onsets = events.dataframe['xmin'] + tmin
        self.offsets = None

        if type(tmax) is str:
            if tmax == 'maxlen':
                tmax = events.dataframe['duration'].max()
                fixed_duration = tmax - tmin
            elif tmax == 'unequal':
                assert 'xmax' in events.dataframe, 'If tmax is uniqual, xmax needs to be provided per event'
                fixed_duration = None
                self.offsets = events.dataframe['xmax']
            else:
                raise NotImplementedError
        elif type(tmax) == int or type(tmax) == float:
            tmax = float(tmax)
            fixed_duration = tmax - tmin
        else:
            raise NotImplementedError

        self.fixed_duration = fixed_duration
        sr = raw.sampling_rate
        for k in self.data_.keys():
            if fixed_duration is not None:
                for ons in self.onsets:
                    self.data_augmented_[k].append([])
                    if sec2ind(ons, sr) < 0 or sec2ind(ons, sr) + sec2ind(fixed_duration, sr) > len(raw.data[k]):
                        warnings.warn('Index out of range, wrapping the array')
                    self.data_[k].append(raw.data[k].take(range(sec2ind(ons, sr),
                                                                sec2ind(ons, sr) + sec2ind(fixed_duration, sr)),
                                                          axis=0, mode='wrap'))
                    for i in range(-9, 10):
                        self.data_augmented_[k][-1].append(raw.data[k].take(range(sec2ind(ons, sr) + i,
                                                                sec2ind(ons, sr) + i + sec2ind(fixed_duration, sr)),
                                                          axis=0, mode='wrap'))
                    self.data_augmented_[k][-1] = np.array(self.data_augmented_[k][-1])
                self.data_[k] = np.array(self.data_[k])
            else:
                for ons, off in zip(self.onsets, self.offsets):
                    self.data_[k].append(raw.data[k][sec2ind(ons, sr):sec2ind(off, sr)])


    def data2array(self):
        assert all([isinstance(d, np.ndarray) for d in self.data_.values()]), \
                                    'Cannot concatenate all data if epochs are not equal size'
        x, aug, names = [], [], []
        for k, v in self.data_.items():
            names.append(k)
            x.append(v)
            aug.append(self.data_augmented_[k])

        x = np.array(x)
        aug = np.array(aug)
        xx = x.transpose((1, 2, 0, 3)) # epochs x timepoints x feature sets x channels
        # xxx = xx.reshape((xx.shape[0], xx.shape[1], -1)) # stack channels and feature sets: Lennart has default order C
        # xxxx = xxx.reshape((xx.shape[0], -1), order='F') # stack over timestamps: Lennart has order F
        aug = aug.transpose((2, 1, 3, 0, -1))
        aug_dims = aug.shape

        self.data = xx
        self.data_augmented = aug.transpose(1, 0, 2, 3, 4).reshape((aug_dims[0]*aug_dims[1],
                                                                    aug_dims[2], aug_dims[3], aug_dims[4]))
        self.feature_names = names

    # def augmented2datadim(self):
    #     dims = self.data.shape
    #     self.data = self.data.transpose(1, 0, 2, 3, 4).reshape((dims[0]*dims[1], dims[2], dims[3], dims[4]))




class TorchDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, index):
        #return torch.Tensor(self.inputs[index]), torch.Tensor(self.outputs[index])
        return self.inputs[index], self.outputs[index]


def objective(trial, x, y):
    params = {
        'kernel_size': trial.suggest_int('kernel_size', 2, 8, step=2),
        'batch_size': trial.suggest_int('batch_size', 4, 24, step=4),
        'n_epochs': trial.suggest_int('n_epochs', 10, 50, step=5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
    }

    # x dimensions: trials x timepoints x bands x channels
    # trainer expects: trials x channels x bands x timepoints
    train_accuracy, validation_accuracy = trainer(np.transpose(x, (0, 3, 2, 1)), y, params)
    print(f'Train accuracy: {train_accuracy}, validation accuracy: {validation_accuracy}')
    trial.set_user_attr("trainAccuracy", train_accuracy)
    return validation_accuracy

def trainer(train_inputs, train_outputs, val_inputs, val_outputs, params):
    # data
    train_dataset = TorchDataset(train_inputs, train_outputs)
    val_dataset = TorchDataset(val_inputs, val_outputs)
    #train_size = round(len(dataset) * .9)
    #trainset, valset = torch.utils.data.random_split(dataset, (train_size, len(dataset) - train_size))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    # set up torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu") # "cuda:0"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(device)
    print('Device: ', device)
    torch.manual_seed(0)

    # set up model
    net = make_model(in_dim=train_inputs.shape[1],
                     out_dim=len(np.unique(train_outputs)),
                     kernel_size=params['kernel_size'])
    net.to(device)

    # optimizer and loss
    adam = torch.optim.Adam(net.parameters(), params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train
    L = np.zeros(params['n_epochs'])
    L_val = np.zeros(params['n_epochs'])
    correct_val = np.zeros(params['n_epochs'])

    for e in tqdm.trange(params['n_epochs']):
        net = net.train()
        correct = 0
        for x, t in train_loader:
            out = net(torch.autograd.Variable(x, requires_grad=False).to(device))
            loss = criterion(out, torch.autograd.Variable(t, requires_grad=False).to(device))
            L[e] += loss.cpu().detach().numpy()
            net.zero_grad()
            loss.backward()
            adam.step()
            correct += (np.argmax(out.cpu().detach().numpy(), 1) == t.detach().numpy()).astype(float).mean()
        L[e] /= len(train_loader)
        correct /= len(train_loader)
        print(f'\ne{e}: train loss: {L[e]},  train accuracy: {correct}')

        net = net.eval()
        for b, (x_val, t_val) in enumerate(val_loader):
            out_val = net(torch.autograd.Variable(x_val, requires_grad=False).to(device))
            loss_val = criterion.cuda()(out_val, torch.autograd.Variable(t_val, requires_grad=False).to(device))
            L_val[e] += loss_val.cpu().detach().numpy()
            correct_val[e] += (np.argmax(out_val.cpu().detach().numpy(), 1) == t_val.detach().numpy()).astype(float).mean()
        L_val[e] /= len(val_loader)
        correct_val[e] /= len(val_loader)
        print(f'e{e}: validation loss: {L_val[e]}, validation accuracy: {correct_val[e]}')

    torch.save({
        'epoch': e,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': adam.state_dict(),
        'loss': L[e],
    }, '/Fridge/users/julia/project_corticom/results/14nav/cc2/cnn/train_8runs/hfb/10Hz_0.0_1.0/sample_model.pth')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return L[-1], L_val[-1]

def tester(inputs, outputs, params):
    # data
    dataset = TorchDataset(inputs, outputs)
    val_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    # set up torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu") # "cuda:0"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(device)
    print('Device: ', device)

    # set up model
    model_path = '/Fridge/users/julia/project_corticom/results/14nav/cc2/cnn/train_8runs/hfb/10Hz_0.0_1.0/sample_model.pth'
    saved = torch.load(model_path, map_location=torch.device(device))
    net = make_model(in_dim=inputs.shape[1], out_dim=len(np.unique(outputs)), kernel_size=params['kernel_size'])
    net.load_state_dict(saved['model_state_dict'])
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    L_val = 0
    correct_val = 0

    net = net.eval()
    for b, (x_val, t_val) in enumerate(val_loader):
        out_val = net(torch.autograd.Variable(x_val, requires_grad=False).to(device))
        loss_val = criterion.cuda()(out_val, torch.autograd.Variable(t_val, requires_grad=False).to(device))
        L_val += loss_val.cpu().detach().numpy()
        correct_val += (np.argmax(out_val.cpu().detach().numpy(), 1) == t_val.detach().numpy()).astype(float).mean()
    L_val /= len(val_loader)
    correct_val /= len(val_loader)
    print(f'test loss: {L_val}, validation accuracy: {correct_val}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return correct_val


def main(config_path):
    config = load_config(Path(config_path))
    params = config['classify']
    tmin = params['epochs']['tmin']
    tmax = params['epochs']['tmax']
    sr = params['sampling_rate']
    features = to_list(params['bands'])
    variant = params['variant']
    runs = [i for i in config['dyn_runs'] if config['order'][i] == variant]

    name = config['subject']
    task = config['task']
    gen_save_path = Path(config['save_path']) / task / name / params['strategy'] / f'train_{len(runs)}runs' / '_'.join(features)
    gen_plot_path = Path(config['plot_path']) / task / name / params['strategy'] / f'train_{len(runs)}runs' / '_'.join(features)
    save_path = gen_save_path / f'{str(sr)}Hz_{str(tmin)}_{str(tmax)}'
    plot_path = gen_plot_path / f'{str(sr)}Hz_{str(tmin)}_{str(tmax)}'
    save_path.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)

    raw = {k: [] for k in runs}
    events = {k: [] for k in runs}
    scaler = {k: [] for k in runs}
    epochs = {k: [] for k in runs}


    #for word in list(task_info['codes'][variant].values())[:-1]:
    #selection = [word, 'rust']
    if variant=='1-7':
        selection = ['links', 'rechts', 'selecteer', 'rust']
    elif variant=='8-14':
        selection =['noord', 'oost', 'zuid', 'west', 'rust']
    else:
        raise NotImplementedError

    for run in runs:
        raw[run] = Dataset(id=name,
                      input_paths={k: v for k, v in config['feature_paths'][sr][run].items() if k in features},
                      channel_paths={k: config['channel_paths'][run] for k in features},
                      sampling_rate=sr)
        events[run] = Events(id=name,
                        events_path=config['events_paths'][run],
                        selection=selection,
                        units='seconds')

        scaler[run] = Scaler(mean_paths={k: v for k, v in config['mean_paths'][sr][run].items() if k in features},
                        scale_paths={k: v for k, v in config['scale_paths'][sr][run].items() if k in features})
        raw_transformed = scaler[run].transform(raw[run])

        epochs[run] = Epochs(raw_transformed, events[run], tmin=tmin, tmax=tmax)
        epochs[run].data2array(use_augmented=True)
        #epochs[run].augmented2datadim()

    # concatenate runs
    events_all = Events(id=name)
    events_all.dataframe = pd.concat([events[i].dataframe for i in runs]).reset_index(drop=True)
    events_all.update_label_encoder()

    epochs_data = np.vstack([epochs[i].data for i in runs])  # will be different per run if maxlen!
    events_data = np.hstack([events[i].data for i in runs])
    epochs_augmented_data = np.vstack([epochs[i].data_augmented for i in runs])

    train_indices, val_indices = train_test_split(range(len(events_all.data)), test_size=0.1, random_state=42)
    train_epochs = epochs_augmented_data[train_indices]
    train_events = events_data[train_indices]
    val_epochs = epochs_data[val_indices]
    val_events = events_data[val_indices]

    # tile train events, reshape train epochs
    events[run].data = np.tile(events[run].data, (epochs[run].data.shape[0], 1)).T.reshape(-1, )
    events[run].classes = np.tile(events[run].classes, (epochs[run].data.shape[0], 1)).T.reshape(-1, )

    # create an optimizer
    from plumpy.ml.optimization import Optimizer
    opt = Optimizer(id=f'cnn_{"_".join(selection)}',
                    args=params['cnn_specs'],
                    obj_fun=lambda trial: objective(trial,
                                                     x_train=train_epochs, y_train=train_events,
                                                     x_val=val_epochs, y_val=val_events),
                    direction='maximize')

    # run optimization
    opt.optimize()
    params = opt.study.best_params

    # test
    runs = [i for i in config['classify']['cnn_specs']['test_runs'] if config['order'][i] == variant]
    raw = {k: [] for k in runs}
    events = {k: [] for k in runs}
    scaler = {k: [] for k in runs}
    epochs = {k: [] for k in runs}

    for run in runs:
        raw[run] = Dataset(id=name,
                           input_paths={k: v for k, v in config['feature_paths'][sr][run].items() if
                                        k in features},
                           channel_paths={k: config['channel_paths'][run] for k in features},
                           sampling_rate=sr)
        events[run] = Events(id=name,
                             events_path=config['events_paths'][run],
                             # selection=params['classes'][variant],
                             selection=selection,
                             units='seconds')

        scaler[run] = Scaler(
            mean_paths={k: v for k, v in config['mean_paths'][sr][run].items() if k in features},
            scale_paths={k: v for k, v in config['scale_paths'][sr][run].items() if k in features})
        raw_transformed = scaler[run].transform(raw[run])

        epochs[run] = Epochs(raw_transformed, events[run], tmin=tmin, tmax=tmax)
        epochs[run].data2array(use_augmented=False)


        tester(np.transpose(epochs[run].data, (0, 3, 2, 1)), events[run].data, params)


if __name__ == '__main__':
    main('/Fridge/users/julia/project_corticom/cc2/config_14nav.yml')

