import os
import glob
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from src.efficient_kan import KAN, KANLinear
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
import math
import seaborn as sns
import networkx as nx
from enum import Enum

np.set_printoptions(threshold = np.inf)

np.set_printoptions(suppress = True)


class BendingDataset(Dataset):
    def __init__(self, root_job_dir, root_cross_section_dir):
        self.root_job_dir = root_job_dir
        self.root_cross_section_dir = root_cross_section_dir
        self.job_files = sorted(glob.glob(os.path.join(root_job_dir, 'job*.csv')))
        self.sequence_length = 6
        self.cross_section_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()
        self.min_max_scaler = MinMaxScaler()
        self.input_scalers = [MinMaxScaler() for _ in range(24)]
        self.fit_scalers()

    def __len__(self):
        return len(self.job_files)

    def fit_scalers(self):
        all_job_data = pd.concat([pd.read_csv(file) for file in self.job_files])

        all_input_data = all_job_data.iloc[:, :24]
        all_labels = all_job_data.iloc[:, 24:27]
        for i in range(24):
            self.input_scalers[i].fit(all_input_data.iloc[:, i].values.reshape(-1, 1))

        self.label_scaler.fit(all_labels.values)

        all_cross_section_data = []
        for file in self.job_files:
            job_id = int(os.path.basename(file)[3:-4])
            for bending_id in range(0, self.get_num_bends(job_id) + 1):
                try:
                    file_path = os.path.join(self.root_cross_section_dir, f'job{job_id}-bending{bending_id}.csv')
                    data = pd.read_csv(file_path).iloc[:]
                    all_cross_section_data.append(data.values)
                except FileNotFoundError:
                    continue

        all_cross_section_data = np.vstack(all_cross_section_data)

        self.cross_section_scaler.fit(all_cross_section_data)
    def construct_bend_edges(self):
        pass
    def get_num_bends(self, job_id):
        file_path = os.path.join(self.root_job_dir, f'job{job_id}.csv')
        data = pd.read_csv(file_path).iloc[:]
        return data.shape[0]

    def load_cross_section_data(self, job_id, bending_id):
        file_path = os.path.join(self.root_cross_section_dir, f'job{job_id}-bending{bending_id}.csv')
        data = pd.read_csv(file_path).iloc[:]
        scaled_data = self.cross_section_scaler.transform(data.values)
        return torch.tensor(scaled_data, dtype=torch.float32)

    def load_job_data(self, job_id):
        file_path = os.path.join(self.root_job_dir, f'job{job_id}.csv')
        data = pd.read_csv(file_path).iloc[:]

        input_data = data.iloc[:, :24]
        input_scaled_columns = [
            self.input_scalers[i].transform(input_data.iloc[:, i].values.reshape(-1, 1))
            for i in range(24)
        ]
        input_scaled = np.hstack(input_scaled_columns)
        input_data = torch.tensor(input_scaled, dtype=torch.float32)

        labels = data.iloc[:, 24:27]
        labels_scaled = self.label_scaler.transform(labels.values)
        labels = torch.tensor(labels_scaled, dtype=torch.float32)

        return input_data, labels

    def __getitem__(self, index):
        file_path = self.job_files[index]
        job_id = int(os.path.basename(file_path)[3:-4])

        input_data, labels = self.load_job_data(job_id)
        num_bends = input_data.shape[0]

        cross_section_data = []
        for bending_id in range(0, num_bends + 1):
            try:
                cs_data = self.load_cross_section_data(job_id, bending_id)
                assert cs_data.shape[0] == self.sequence_length, "Sequence length does not match the expected length."
                cross_section_data.append(cs_data)
            except FileNotFoundError:
                continue
        encoder_inputs = [input_data[i] for i in range(num_bends)]
        cross_section_labels = cross_section_data
        axial_labels = labels

        return encoder_inputs, cross_section_labels, axial_labels, num_bends

    def inverse_transform_axial(self, labels):
        return self.label_scaler.inverse_transform(labels)

    def inverse_transform_cross_section(self, cross_section_data):
        if len(cross_section_data.shape) == 1:
            cross_section_data = cross_section_data.reshape(-1, 24)
        inverse_data = self.cross_section_scaler.inverse_transform(cross_section_data)
        return inverse_data

start_time = time.time()
root_job_dir = r'.\data\job'
root_cross_section_dir = r'.\data\cross_section'
dataset = BendingDataset(root_job_dir, root_cross_section_dir)
def custom_collate_fn(batch):
    encoder_inputs, lstm_labels, original_labels, num_bends = zip(*batch)
    max_num_bends = max(num_bends)
    padded_encoder_inputs = []
    padded_lstm_labels = []
    padded_original_labels = []

    for inputs, labels, orig_labels, n_bends in zip(encoder_inputs, lstm_labels, original_labels, num_bends):
        padded_inputs = list(inputs)
        padded_inputs.extend([torch.zeros_like(inputs[0])] * (max_num_bends - n_bends))
        padded_encoder_inputs.append(padded_inputs)
        padded_labels = list(labels)
        padded_labels.extend([torch.zeros_like(labels[0])] * (max_num_bends - n_bends))
        padded_lstm_labels.append(padded_labels)
        padded_orig_labels = list(orig_labels)
        padded_orig_labels.extend([torch.zeros_like(orig_labels[0])] * (max_num_bends - n_bends))
        padded_original_labels.append(padded_orig_labels)

    encoder_inputs_tensor = torch.stack([torch.stack(inputs) for inputs in padded_encoder_inputs])
    lstm_labels_tensor = torch.stack([torch.stack(labels) for labels in padded_lstm_labels])
    original_labels_tensor = torch.stack([torch.stack(labels) for labels in padded_original_labels])

    return encoder_inputs_tensor, lstm_labels_tensor, original_labels_tensor, max_num_bends


# seed_value = 21
# random.seed(seed_value)
# np.random.seed(seed_value)
# torch.manual_seed(seed_value)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

dataloader = DataLoader(dataset, batch_size=96, shuffle=True, collate_fn=custom_collate_fn)
train_ratio = 0.8
val_ratio = 1 - train_ratio

dataset_length = len(dataset)

train_length = int(train_ratio * dataset_length)
val_length = dataset_length - train_length

#train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(seed_value))#random seed
train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, collate_fn=custom_collate_fn)

def compute_transformer_loss(predicted, actual, input_data, loss_function, distance_column=18, ):
    distances = input_data[:, :, distance_column]
    distances_to_start = torch.cumsum(distances, dim=1)
    position_errors = torch.norm(predicted - actual, dim=-1)
    remaining_position_errors = position_errors[:, 1:]
    remaining_distances_to_start = distances_to_start[:, 1:]
    normalized_errors = remaining_position_errors / (remaining_distances_to_start + 1e-8)
    if loss_function == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif loss_function == 'mae':
        loss_fn = nn.L1Loss(reduction='none')
    elif loss_function == 'smooth_l1':
        loss_fn = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
    first_actual_position = actual[:, 0, :]
    first_predicted_position = predicted[:, 0, :]
    first_absolute_error = torch.norm(first_predicted_position - first_actual_position, dim=-1)
    first_absolute_loss = loss_fn(first_absolute_error, torch.zeros_like(first_absolute_error))
    normalized_loss = loss_fn(normalized_errors, torch.zeros_like(normalized_errors))
    combined_losses = torch.cat([first_absolute_loss.unsqueeze(1), normalized_loss], dim=1)
    loss = combined_losses.mean()
    return loss
class BiGRU1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=240):
        super(BiGRU1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self, x , hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        output, h_n = self.gru(x,hidden)
        initial_output = output[:, -1, :]
        outputs = [self.fc(initial_output)]

        for _ in range(5):
            output, h_n = self.gru(x, h_n)
            output = self.fc(output.squeeze(1))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        forward_hidden_state_last = h_n[num_layers - 1]
        backward_hidden_state_last = h_n[-1]
        gru_hidden_state_last = torch.cat((forward_hidden_state_last, backward_hidden_state_last), dim=1)
        return outputs, gru_hidden_state_last, h_n

class BiGRU2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=24):
        super(BiGRU2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.kan = KAN([hidden_size * 2,64,output_size])
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.leakyrelu = nn.LeakyReLU()

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self, x, hidden):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        output, h_n = self.gru(x)
        initial_output = output[:, -1, :]
        outputs = [self.leakyrelu(self.fc(initial_output))]

        for _ in range(5):
            output, h_n = self.gru(x, h_n)
            output = self.leakyrelu(self.fc(output.squeeze(1)))
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        forward_hidden_state_last = h_n[num_layers - 1]
        backward_hidden_state_last = h_n[-1]

        gru_hidden_state_last = torch.cat((forward_hidden_state_last, backward_hidden_state_last), dim=1)
        return outputs, gru_hidden_state_last , h_n




class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        output_attentions: bool = True,
        attention_probs_dropout_prob: float = 0.1,
        use_dense_layer: bool = True,
        use_layer_norm: bool = False,
        mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM,
    ):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.output_attentions = output_attentions
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.mixing_initialization = mixing_initialization
        self.use_dense_layer = use_dense_layer
        self.use_layer_norm = use_layer_norm

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.content_bias = nn.Linear(d_model, nhead, bias=False)
        self.mixing = self.init_mixing_matrix()

        self.dense = nn.Linear(d_model, d_model) if use_dense_layer else nn.Sequential()
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        attention_mask=None,
        head_mask=None,
    ):
        Q = self.Wq(x).view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)
        assert self.mixing.shape == (self.nhead, self.head_dim), f"Mixing matrix shape mismatch: {self.mixing.shape} != ({self.nhead}, {self.head_dim})"
        mixed_Q = Q * self.mixing.unsqueeze(0).unsqueeze(-2)
        attn_scores = torch.matmul(mixed_Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        content_bias = self.content_bias(x).transpose(1, 2).unsqueeze(-2)
        attn_scores += content_bias
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        attn_output = self.dense(attn_output)
        if self.use_layer_norm:
            attn_output = self.layer_norm(x + attn_output)
        if self.output_attentions:
            return (attn_output, attn_weights)
        else:
            return (attn_output,)

    def init_mixing_matrix(self, scale=0.2):
        mixing = torch.zeros(self.nhead, self.head_dim)

        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            dim_head = int(math.ceil(self.head_dim / self.nhead))
            for i in range(self.nhead):
                mixing[i, i * dim_head : (i + 1) * dim_head] = 1.0
        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.fill_(1.0)
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
        else:
            raise ValueError(
                "Unknown mixing matrix initialization: {}".format(
                    self.mixing_initialization
                )
            )
        return nn.Parameter(mixing)

def drop_edge(edge_index, p):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    return edge_index[:, mask]

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_edge_prob, edge_dim=4):
        super(GNNLayer, self).__init__()
        self.gat_b = GATConv(
            in_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=drop_edge_prob,
            edge_dim=edge_dim
        )
        self.gat_s = GATConv(
            in_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=drop_edge_prob,
            edge_dim=edge_dim
        )
        self.drop_edge_prob = drop_edge_prob
        self.weights = nn.Parameter(torch.ones(2))
        self.edge_encoder = nn.Linear(edge_dim, edge_dim)

    def forward(self, x, section_edge_index, bend_edge_index, section_edge_attr, bend_edge_attr):
        batch_size, num_sections, num_nodes_per_section, num_features = x.shape
        encoded_bend_attr = self.edge_encoder(bend_edge_attr)
        encoded_section_attr = self.edge_encoder(section_edge_attr)
        x_b = x.view(batch_size, -1, num_features)
        bend_data_list = []
        for i in range(batch_size):
            bend_attr_i = encoded_bend_attr[i,:,:]
            bend_data = Data(
                x=x_b[i],
                edge_index=bend_edge_index,
                edge_attr=bend_attr_i
            )
            bend_data_list.append(bend_data)

        batched_data_b = Batch.from_data_list(bend_data_list)
        if self.training and self.drop_edge_prob > 0:
            batched_data_b.edge_index, _ = drop_edge(batched_data_b.edge_index, p=self.drop_edge_prob)

        x_b_batched = self.gat_b(
            batched_data_b.x,
            batched_data_b.edge_index,
            edge_attr=batched_data_b.edge_attr
        )
        x_b_batched = F.leaky_relu_(x_b_batched)
        x_b_out = x_b_batched.view(batch_size, -1, self.gat_b.out_channels)

        x_s_out = []
        for sec_idx in range(num_sections):
            section_data_list = []
            x_s = x[:, sec_idx, :, :]
            for i in range(batch_size):
                section_attr_i = encoded_section_attr[i,:,:]
                section_data = Data(
                    x=x_s[i],
                    edge_index=section_edge_index,
                    edge_attr=section_attr_i
                )
                section_data_list.append(section_data)
            batched_data_s = Batch.from_data_list(section_data_list)
            if self.training and self.drop_edge_prob > 0:
                batched_data_s.edge_index, _ = drop_edge(batched_data_s.edge_index, p=self.drop_edge_prob)
            x_sec = self.gat_s(
                batched_data_s.x,
                batched_data_s.edge_index,
                edge_attr=batched_data_s.edge_attr
            )
            x_s_out.append(x_sec.view(batch_size, num_nodes_per_section, -1))

        x_s_out = torch.stack(x_s_out, dim=1).view(batch_size, -1, self.gat_s.out_channels)
        weights = F.softmax(self.weights, dim=0)
        return weights[0] * x_b_out + weights[1] * x_s_out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, output_size)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.layer1(x))
        x = self.layer2(x)
        return x


def edge_index_to_adj_matrix(edge_index, num_nodes):
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix[edge_index[1], edge_index[0]] = 1
    return adjacency_matrix

class BendingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, d_model, nhead, factor, output_size):
        super(BendingModel, self).__init__()
        self.gru1 = BiGRU1(input_size, hidden_size, num_layers)
        self.gru2 = BiGRU2(168, hidden_size, num_layers)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            output_attentions=True,
            attention_probs_dropout_prob=0.1,
            use_dense_layer=True,
            use_layer_norm=True,
            mixing_initialization=MixingMatrixInit.UNIFORM
        )
        self.gnn_layer1 = GNNLayer(10, 10, drop_edge_prob=0, edge_dim=4)
        self.section_mlps = nn.ModuleList([MLP(10,10,1) for _ in range(6)])
        self.mlp = MLP(d_model, 50, output_size)
        self.fc_adjust = nn.Linear(168, d_model)
        self.section_edge_index, self.section_edge_attr = self.construct_section_edge_index(24, device=device)
        self.bend_edge_index, self.bend_edge_attr = self.construct_bend_edge_index(24, 6,device=device)


    def forward(self, x, num_bends):
        batch_size, _, input_size = x.size()
        gru_outputs = []
        h_n = None
        for bend in range(num_bends):
            input_data = x[:, bend, :]
            gru_output, _, h_n= self.gru1(input_data.unsqueeze(1), h_n)

            gru_output = gru_output.squeeze(1)
            gru_outputs.append(gru_output)

        gru_outputs = torch.stack(gru_outputs, dim=1)
        num_sections = 6
        gru_output_for_gnn_inputs = gru_outputs.view(batch_size, num_bends, num_sections, 24, 10)

        gnn_outputs = []
        gnn_outputs_mlp = []
        for bend in range(num_bends):
            bend_gnn_outputs = []
            bend_gnn_outputs_mlp = []

            gnn_input = gru_output_for_gnn_inputs[:, bend]

            section_edge_index = self.section_edge_index.to(x.device)
            bend_edge_index = self.bend_edge_index.to(x.device)
            bend_edge_attr_batch = self.bend_edge_attr.unsqueeze(0).repeat(batch_size, 1, 1)
            section_edge_attr_batch = self.section_edge_attr.unsqueeze(0).repeat(batch_size, 1, 1)
            gnn_output = self.gnn_layer1(
                gnn_input,
                section_edge_index=section_edge_index,
                bend_edge_index=bend_edge_index,
                section_edge_attr=section_edge_attr_batch,
                bend_edge_attr=bend_edge_attr_batch
            )

            gnn_output_mlp = []
            for section in range(num_sections):
                section_output = gnn_output[:, section * 24:(section + 1) * 24].view(batch_size, 24,
                                                                                     10)

                section_output = self.section_mlps[section](section_output)
                gnn_output_mlp.append(section_output)

            gnn_output_mlp = torch.stack(gnn_output_mlp,dim=1)


            bend_gnn_outputs.append(gnn_output)
            bend_gnn_outputs_mlp.append(gnn_output_mlp)

            gnn_outputs.append(bend_gnn_outputs)
            gnn_outputs_mlp.append(bend_gnn_outputs_mlp)

        gnn_outputs_mlp = torch.stack([item[0] for item in gnn_outputs_mlp],dim=1)
        gnn_outputs_mlp_flattened = gnn_outputs_mlp.view(batch_size, num_bends, num_sections * 24)

        gru2_outputs = []
        gru2_hidden_state_lasts = []
        h_n2 = None
        for bend in range(num_bends):
            input_data = torch.cat((gnn_outputs_mlp_flattened[:, bend, :], x[:, bend, :]), dim=1)
            gru2_output, gru_hidden_state_last, h_n2= self.gru2(input_data.unsqueeze(1), h_n2)
            gru2_output = gru2_output.squeeze(1)
            gru2_outputs.append(gru2_output)
            gru2_hidden_state_lasts.append(gru_hidden_state_last)

        gru2_outputs = torch.stack(gru2_outputs, dim=1)
        gru2_outputs_flattened = gru2_outputs.view(gru2_outputs.shape[0], 7, -1)
        gru2_outputs_residual = gru2_outputs_flattened + gnn_outputs_mlp_flattened
        transformer_inputs = []
        for bend in range(num_bends):
            transformer_input = torch.cat((gru2_outputs_residual[:, bend, :], x[:, bend, :]), dim=1)
            transformer_input = self.fc_adjust(transformer_input)
            transformer_inputs.append(transformer_input)

        transformer_inputs = torch.stack(transformer_inputs)
        transformer_inputs = transformer_inputs.permute(1, 0, 2)
        transformer_output, _ = self.multi_head_attention(transformer_inputs)
        output = self.mlp(transformer_output)
        section_predictions = gnn_outputs_mlp.squeeze(-1)
        return output, section_predictions

    def construct_bend_edge_index(self, num_nodes_per_section, num_sections, device):
        edges = []
        edge_attrs = []
        TYPE_MAP = {
            'intra_front_ring': [1, 0, 0, 0],
            'intra_back_ring': [0, 1, 0, 0],
            'intra_cross': [0, 0, 1, 0],
            'inter_section': [0, 0, 0, 1]
        }

        for section in range(num_sections):
            start_node = section * num_nodes_per_section
            end_node = (section + 1) * num_nodes_per_section
            half = num_nodes_per_section // 2

            for i in range(start_node, start_node + half):
                next_idx = start_node + (i - start_node + 1) % half
                edges.append([i, next_idx])
                edge_attrs.append(TYPE_MAP['intra_front_ring'])

            for j in range(start_node + half, end_node):
                next_idx = start_node + half + (j - start_node - half + 1) % half
                edges.append([j, next_idx])
                edge_attrs.append(TYPE_MAP['intra_back_ring'])

            for k in range(start_node, start_node + half):
                paired_node = k + half
                edges.append([k, paired_node])
                edge_attrs.append(TYPE_MAP['intra_cross'])

        for section in range(num_sections - 1):
            for i in range(num_nodes_per_section):
                current_node = section * num_nodes_per_section + i
                next_node = (section + 1) * num_nodes_per_section + i
                edges.append([current_node, next_node])
                attr = TYPE_MAP['inter_section']
                edge_attrs.append(attr)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).to(device)

        return edge_index, edge_attr

    def construct_section_edge_index(self, num_nodes, device):
        edges = []
        edge_attrs = []
        half = num_nodes // 2
        TYPE_MAP = {
            'intra_front': [1, 0, 0, 0],
            'intra_back': [0, 1, 0, 0],
            'inter_halves': [0, 0, 1, 0],
            'cross_section': [0, 0, 0, 1]
        }

        for i in range(half):
            next_index = (i + 1) % half
            edges.append([i, next_index])
            edge_attrs.append(TYPE_MAP['intra_front'])

        for j in range(half, num_nodes):
            next_index = (j + 1) % num_nodes
            if next_index == half:
                next_index = num_nodes
            edges.append([j, next_index])
            edge_attrs.append(TYPE_MAP['intra_back'])

        for k in range(half):
            edges.append([k, k + half])
            edge_attrs.append(TYPE_MAP['inter_halves'])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float).to(device)
        return edge_index,  edge_attrs



input_size = 24
hidden_size = 56
num_layers = 2
d_model = 170
nhead = 5
output_size = 3
factor = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_runs = 1
num_epochs = 500


final_results = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")

    model = BendingModel(input_size, hidden_size, num_layers, d_model, nhead, factor, output_size).to(device)
    criterion = nn.MSELoss()

    transformer_params = (list(model.fc_adjust.parameters()) + list(model.multi_head_attention.parameters()) + list(model.mlp.parameters())+list(model.gru2.parameters()))
    section_params = list(model.gru1.parameters()) + list(model.gnn_layer1.parameters()) + list(model.section_mlps.parameters())

    optimizer_transformer = torch.optim.Adam(transformer_params, lr=0.001, weight_decay=5e-8)
    optimizer_section = torch.optim.Adam(section_params, lr=0.001, weight_decay=5e-8)

    train_transformer_loss_history = []
    train_section_loss_history = []
    train_total_loss_history = []
    val_transformer_loss_history = []
    val_section_loss_history = []
    val_total_loss_history = []

    start_time = time.time()


    for epoch in range(num_epochs):
        model.train()
        total_train_transformer_loss = 0
        total_train_section_loss = 0
        total_train_loss = 0

        for batch_idx, (encoder_inputs, section_labels, original_labels, max_num_bends) in enumerate(train_dataloader):
            encoder_inputs = encoder_inputs.to(device)
            section_labels = section_labels.to(device)
            original_labels = original_labels.to(device)
            transformer_outputs, section_predictions = model(encoder_inputs, max_num_bends)
            transformer_loss = compute_transformer_loss(transformer_outputs, original_labels, input_data=encoder_inputs, distance_column=18, loss_function='smooth_l1')
            mask = section_labels != 0
            section_labels_filtered = section_labels[mask]
            section_predictions_filtered = section_predictions[mask]
            section_loss = criterion(section_predictions_filtered, section_labels_filtered)
            loss = transformer_loss + section_loss
            optimizer_transformer.zero_grad()
            transformer_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(transformer_params, max_norm=0.1)
            optimizer_transformer.step()
            optimizer_section.zero_grad()
            section_loss.backward()
            optimizer_section.step()
            total_train_transformer_loss += transformer_loss.item()
            total_train_section_loss += section_loss.item()
            total_train_loss += loss.item()

        average_train_transformer_loss = total_train_transformer_loss / (batch_idx + 1)
        average_train_section_loss = total_train_section_loss / (batch_idx + 1)
        average_train_loss = total_train_loss / (batch_idx + 1)
        train_transformer_loss_history.append(average_train_transformer_loss)
        train_section_loss_history.append(average_train_section_loss)
        train_total_loss_history.append(average_train_loss)

        model.eval()
        total_val_transformer_loss = 0
        total_val_section_loss = 0
        total_val_loss = 0
        all_val_original_labels = []
        all_val_transformer_outputs = []
        all_val_section_labels = []
        all_val_section_predictions = []

        with torch.no_grad():
            for batch_idx, (encoder_inputs, section_labels, original_labels, max_num_bends) in enumerate(
                    val_dataloader):
                encoder_inputs = encoder_inputs.to(device)
                section_labels = section_labels.to(device)
                original_labels = original_labels.to(device)
                transformer_outputs, section_predictions = model(encoder_inputs, max_num_bends)
                transformer_loss = criterion(original_labels, transformer_outputs)
                mask = section_labels != 0
                section_labels_filtered = section_labels[mask]
                section_predictions_filtered = section_predictions[mask]
                section_loss = criterion(section_predictions_filtered, section_labels_filtered)
                loss = transformer_loss + section_loss
                total_val_transformer_loss += transformer_loss.item()
                total_val_section_loss += section_loss.item()
                total_val_loss += loss.item()
                all_val_original_labels.append(original_labels.cpu().numpy())
                all_val_transformer_outputs.append(transformer_outputs.cpu().numpy())
                all_val_section_labels.append(section_labels.cpu().numpy())
                all_val_section_predictions.append(section_predictions.cpu().numpy())
        average_val_transformer_loss = total_val_transformer_loss / (batch_idx + 1)
        average_val_section_loss = total_val_section_loss / (batch_idx + 1)
        average_val_loss = total_val_loss / (batch_idx + 1)
        val_transformer_loss_history.append(average_val_transformer_loss)
        val_section_loss_history.append(average_val_section_loss)
        val_total_loss_history.append(average_val_loss)

        all_val_original_labels = np.concatenate(all_val_original_labels, axis=0).reshape(-1, all_val_original_labels[
            0].shape[-1])

        all_val_transformer_outputs = np.concatenate(all_val_transformer_outputs, axis=0).reshape(-1,
                                                                                                  all_val_transformer_outputs[
                                                                                                      0].shape[-1])
        all_val_section_labels = np.concatenate(all_val_section_labels, axis=0).reshape(-1,
                                                                                        all_val_section_labels[0].shape[
                                                                                            -1])
        all_val_section_predictions = np.concatenate(all_val_section_predictions, axis=0).reshape(-1,
                                                                                                  all_val_section_predictions[
                                                                                                      0].shape[-1])

        section_mask = all_val_section_labels != 0
        axial_mask = all_val_original_labels != 0

        all_val_section_labels_filtered = all_val_section_labels[section_mask]
        all_val_section_predictions_filtered = all_val_section_predictions[section_mask]
        all_val_original_labels_filtered = all_val_original_labels[axial_mask]
        all_val_transformer_outputs_filtered = all_val_transformer_outputs[axial_mask]

        all_val_original_labels = dataset.inverse_transform_axial(all_val_original_labels)
        all_val_transformer_outputs = dataset.inverse_transform_axial(all_val_transformer_outputs)
        all_val_section_labels_filtered = dataset.inverse_transform_cross_section(all_val_section_labels_filtered)
        all_val_section_predictions_filtered = dataset.inverse_transform_cross_section(all_val_section_predictions_filtered)
        transformer_mape = mean_absolute_percentage_error(all_val_original_labels_filtered,
                                                          all_val_transformer_outputs_filtered)
        transformer_mae = mean_absolute_error(all_val_original_labels, all_val_transformer_outputs)
        transformer_mse = mean_squared_error(all_val_original_labels, all_val_transformer_outputs, squared=True)
        transformer_rmse = mean_squared_error(all_val_original_labels, all_val_transformer_outputs, squared=False)
        transformer_r2 = r2_score(all_val_original_labels, all_val_transformer_outputs)

        section_mape = mean_absolute_percentage_error(all_val_section_labels_filtered,
                                                      all_val_section_predictions_filtered)
        section_mae = mean_absolute_error(all_val_section_labels_filtered, all_val_section_predictions_filtered)
        section_rmse = mean_squared_error(all_val_section_labels_filtered, all_val_section_predictions_filtered,
                                          squared=False)
        section_r2 = r2_score(all_val_section_labels_filtered, all_val_section_predictions_filtered)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Transformer Loss: {average_train_transformer_loss:.4f}, Train Section Loss: {average_train_section_loss:.4f}, Train Total Loss: {average_train_loss:.4f}, "
              f"Val Transformer Loss: {average_val_transformer_loss:.4f}, Val Section Loss: {average_val_section_loss:.4f}, Val Total Loss: {average_val_loss:.4f}, "
              f"Transformer MAPE: {transformer_mape:.4f}, Transformer MAE: {transformer_mae:.4f}, Transformer RMSE: {transformer_rmse:.4f}, Transformer R2: {transformer_r2:.4f}, "
              f"Section MAPE: {section_mape:.4f},Section MAE: {section_mae:.4f}, Section RMSE: {section_rmse:.4f}, Section R2: {section_r2:.4f}")

    end_time = time.time()
    print(f"Training time for Run {run + 1}: {end_time - start_time:.2f} seconds")

    final_results.append({
        'run': run + 1,
        'train_transformer_loss': average_train_transformer_loss,
        'train_section_loss': average_train_section_loss,
        'train_total_loss': average_train_loss,
        'val_transformer_loss': average_val_transformer_loss,
        'val_section_loss': average_val_section_loss,
        'val_total_loss': average_val_loss,
        'transformer_mape': transformer_mape,
        'transformer_mae': transformer_mae,
        'transformer_rmse': transformer_rmse,
        'transformer_r2': transformer_r2,
        'section_mape': section_mape,
        'section_mae': section_mae,
        'section_rmse': section_rmse,
        'section_r2': section_r2
    })

print("\nFinal Results:")
for result in final_results:
    print(f"  Val Transformer Loss: {result['val_transformer_loss']:.4f}")

for result in final_results:
    print(f"  Val Section Loss: {result['val_section_loss']:.4f}")

for result in final_results:
    print(f"  Val Total Loss: {result['val_total_loss']:.4f}")

for result in final_results:
    print(f"  Transformer MAPE: {result['transformer_mape']:.4f}")

for result in final_results:
    print(f"  Transformer MAE: {result['transformer_mae']:.4f}")

for result in final_results:
    print(f"  Transformer RMSE: {result['transformer_rmse']:.4f}")

for result in final_results:
    print(f"  Transformer R2: {result['transformer_r2']:.4f}")

for result in final_results:
    print(f"  Section MAPE: {result['section_mape']:.4f}")

for result in final_results:
    print(f"  Section MAE: {result['section_mae']:.4f}")

for result in final_results:
    print(f"  Section RMSE: {result['section_rmse']:.4f}")

for result in final_results:
    print(f"  Section R2: {result['section_r2']:.4f}")

print("Training complete.")