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

TODO: figure out a way to pass params to optuna objective. through a class only?

python ./train_decoders/optuna_decode_all_hfb_mel.py \
    -i /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_hfb_jip_janneke_car_70-170_25Hz.npy \
    -o /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_audio_jip_janneke_clean_nfft882_hop882_mel40_25Hz.npy \
    --input_sr 25. \
    --output_sr 25. \
    --input_ref car \
    --input_band 70-170 \
    --input_mean /Fridge/users/julia/project_decoding_jip_janneke/data//subject1_hfb_jip_janneke_car_70-170_25Hz_rest_precomputed_mean.npy \
    --input_std /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_hfb_jip_janneke_car_70-170_25Hz_rest_precomputed_std.npy \
    --output_mean /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_audio_jip_janneke_clean_nfft882_hop882_mel40_25Hz_mean.npy \
    --output_std /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_audio_jip_janneke_clean_nfft882_hop882_mel40_25Hz_std.npy \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/old/subject1/plot_model \
    --fragments /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_contexts_jip_janneke_25Hz_step0.04s_window0.36s_speech_only.csv \
    --fragment_len .36 \
    --model_type mlp \
    --mlp_n_blocks 3 \
    --mlp_n_hidden 128 \
    --drop_ratio .0 \
    --n_epochs 1 \
    --learning_rate 8e-4

    --no-dense_bottleneck
    --dense_reduce .7
    --dense_n_layers 10
    --dense_growth_rate 40

    --seq_n_enc_layers 1
    --seq_n_dec_layers 1
    --seq_bidirectional
"""


import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import tqdm
import argparse
import os.path

from plumpy.ml.models import make_model
from torch.utils.data import Dataset, DataLoader

class TorchDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.length = inputs.shape[0]
    def __getitem__(self, index):
        return torch.Tensor(self.inputs[index]), torch.Tensor(self.outputs[index])

def trainer(x, y, params):

    # data
    dataset = TorchDataset(x, y)
    train_size = round(dataset.length * .9)
    trainset, valset = torch.utils.data.random_split(dataset, (train_size, dataset.length - train_size))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=0, drop_last=drop_last)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    # set up torch
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu") # "cuda:0"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(device)
    print('Device: ', device)
    torch.manual_seed(0)

    # set up model
    net = make_model(params, device)

    # optimizer and loss
    adam = torch.optim.Adam(net.parameters(), args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train
    L = np.zeros(params.n_epochs)
    L_val = np.zeros(params.n_epochs)
    min_val_loss = 10000
    early_stop_counter = -1

    for e in tqdm.trange(params.n_epochs):
        net = net.train()
        for (x, t, ith) in train_loader:
            out = net(torch.autograd.Variable(x, requires_grad=False).to(device))
            loss = criterion(out, torch.autograd.Variable(t, requires_grad=False).to(device))
            L[e] += loss.cpu().detach().numpy()
            #print(loss)
            net.zero_grad()
            loss.backward()
            adam.step()
        L[e] /= len(train_loader)

        net = net.eval()
        predicted_val_all = []
        targets_val_all = []
        for b, (x_val, t_val, ith_val) in enumerate(val_loader):
            out_val = net(torch.autograd.Variable(x_val, requires_grad=False).to(device))
            loss_val = criterion.cuda()(out_val, torch.autograd.Variable(t_val, requires_grad=False).to(device))
            L_val[e] += loss_val.cpu().detach().numpy()
            #print('Validation loss: ', loss_val)

            predicted_val_all.append(out_val.detach().cpu().detach().numpy())
            targets_val_all.append(t_val.cpu().detach().numpy())

        L_val[e] /= len(val_loader)
        if L_val[e] < min_val_loss:
            min_val_loss = L_val[e]
            early_stop_counter = -1

        early_stop_counter += 1
        if early_stop_counter >= args.early_stop_max:
            args.n_epochs = e
            L = L[:args.n_epochs]
            L_val = L_val[:args.n_epochs]
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return L_val[-1]



##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural decoding Vanilla')

    # model
    parser.add_argument('--model_type', type=str, choices=['densenet', 'seq2seq', 'mlp', 'resnet'])
    parser.add_argument('--drop_ratio', type=float, default=0.0)

    parser.add_argument('--mlp_n_blocks', type=int, default=3)
    parser.add_argument('--mlp_n_hidden', type=int, default=64)

    parser.add_argument('--dense_bottleneck', dest='dense_bottleneck', action='store_true')
    parser.add_argument('--no-dense_bottleneck', dest='dense_bottleneck', action='store_false')
    parser.add_argument('--dense_reduce', type=float, default=1.)
    parser.add_argument('--dense_n_layers', type=int, default=10)
    parser.add_argument('--dense_growth_rate', type=int, default=10)

    parser.add_argument('--seq_n_enc_layers', type=int, default=1)
    parser.add_argument('--seq_n_dec_layers', type=int, default=1)
    parser.add_argument('--seq_bidirectional', dest='seq_bidirectional', action='store_true')
    parser.add_argument('--no-seq_bidirectional', dest='seq_bidirectional', action='store_false')

    # data
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--input_sr', '-isr', type=float)
    parser.add_argument('--input_ref', type=str)
    parser.add_argument('--input_band', type=str)
    parser.add_argument('--input_mean', '-im', type=str)
    parser.add_argument('--input_std', '-is', type=str)
    parser.add_argument('--input_delay', type=float, default=0.0)
    parser.add_argument('--n_pcs', type=int, default=100)
    parser.add_argument('--use_pca', dest='use_pca', action='store_true')
    parser.add_argument('--no-use_pca', dest='use_pca', action='store_false')

    parser.add_argument('--output_file', '-o', type=str)
    parser.add_argument('--output_sr', '-osr', type=float)
    parser.add_argument('--output_mean', '-om', type=str)
    parser.add_argument('--output_std', '-os', type=str)

    parser.add_argument('--fragments', type=str)
    parser.add_argument('--fragment_len', type=float)

    # training
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--early_stop_max', type=int, default=10000)
    parser.add_argument('--clip_x_value', type=int, default=3)
    parser.add_argument('--clip_t_value', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)

    # other
    parser.add_argument('--save_dir', '-s', type=str)
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.add_argument('--make_plots', dest='make_plots', action='store_true')
    parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')
    parser.add_argument('--save_plots', dest='save_plots', action='store_true')
    parser.add_argument('--no-save_plots', dest='save_plots', action='store_false')

    # binary defaults
    parser.set_defaults(dense_bottleneck=False)
    parser.set_defaults(seq_bidirectional=False)
    parser.set_defaults(use_pca=False)
    parser.set_defaults(save_checkpoints=True)
    parser.set_defaults(make_plots=True)
    parser.set_defaults(save_plots=True)
    args = parser.parse_args()

    trainer(args)


## gradient viz
# after loss.backward()
#
# ave_grads = []
# layers = []
# for n, p in net.named_parameters():
#     if (p.requires_grad) and ("bias" not in n):
#         layers.append(n)
#         ave_grads.append(p.grad.abs().mean())
# plt.plot(ave_grads, alpha=0.3, color="b")
# plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
# plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
# plt.xlim(xmin=0, xmax=len(ave_grads))
# plt.xlabel("Layers")
# plt.ylabel("average gradient")
# plt.title("Gradient flow")
# plt.grid(True)

