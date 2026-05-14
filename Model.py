# ============================================================
# Imports & Configurations
# ============================================================
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import radians, sin, cos, sqrt, asin
import collections
import os
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

# Try to import cartopy for geographic plotting (optional)
try:
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False


DATA_PATH =             # Path to the data file
WINDOW_SIZE = 8                       # Time window size for input sequence
PREDICTION_HORIZON = 1                # Predict the n-th future time point
INPUT_FEATURES =                    # Number of input features
OUTPUT_FEATURES =                    # Number of output features (Longitude, Latitude)
BATCH_SIZE =                        # Training batch size
EPOCHS =                          # Number of training epochs
LEARNING_RATE =                 # Learning rate
TEST_SPLIT_SIZE = 0.1                 # Proportion of test set (10%)
VALIDATION_SPLIT_SIZE = 0.1           # Proportion of validation set (10%)
RANDOM_STATE = 42                     # Random seed for reproducible results

# Early stopping parameters
EARLY_STOPPING_PATIENCE =          # Early stopping patience: max epochs without validation loss improvement
EARLY_STOPPING_MIN_DELTA =       # Minimum improvement threshold: improvement less than this is considered ineffective

# Plotting settings
PLOTS_SAVE_DIR = "plots"             # Directory to save path plots

# Hyperparameters configuration
NOISE_DIM =                        # Dimension of the noise vector
D_HIDDEN_DIM =                    # Discriminator hidden dimension

# ConvLSTM model parameters
CONVLSTM_HIDDEN_DIM = 
CONVLSTM_KERNEL_SIZE = 
CONVLSTM_LAYERS = 
LAMBDA_GAN =                     # GAN loss weight
LAMBDA_L2 =                      # L2 loss weight

# Reshape features into a 2D grid height (rows). e.g., 30 features = 3 x 10.
# Note: INPUT_FEATURES % FEATURE_GRID_H == 0 must be satisfied.
FEATURE_GRID_H = 

# Model save path (default)
MODEL_SAVE_PATH = "best_generator.pth"

# ============================================================
# Utilities, Evaluation & Plotting
# ============================================================

# Logger Class
class Logger(object):
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        # If no filename is specified, use current time
        if filename is None:
            from datetime import datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{current_time}.txt"
        # Open file with utf-8 encoding
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is necessary for Python 3 compatibility.
        # It handles print() function calls with flush command.
        self.terminal.flush()
        self.log.flush()

# Evaluation function
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on the Earth (in km).
    """
    R = 6371  # Earth's average radius (km)
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c


# Plotting function
def plot_top_bottom_typhoons(sorted_errors, results_by_id, num_to_plot=10,
                             all_data=None, all_lengths=None, id_to_indices=None):
    """
    Plot the paths of N typhoons, with the largest and smallest errors.
    If Cartopy is installed, use maps, otherwise use regular plots.

    Parameters:
      - sorted_errors: List sorted by average distance error, each element is { 'id': str, 'avg_dist_error': float }
      - results_by_id: dict[str, {'true': List[[lat,lon]], 'pred': List[[lat,lon]]}]
      - num_to_plot: Number of typhoons to plot for each category (best/worst)
      - all_data, all_lengths, id_to_indices: Optional, for overlaying full paths
    """
    if not sorted_errors:
        print("No typhoon data available for plotting.")
        return

    # Create a subdirectory with current time
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_save_dir = os.path.join(PLOTS_SAVE_DIR, current_time)
    os.makedirs(plots_save_dir, exist_ok=True)
    print(f"\nPlotting typhoon tracks. Plots will be saved to '{plots_save_dir}' directory...")

    best_typhoons = sorted_errors[:num_to_plot]      # Best performance, smallest error
    worst_typhoons = sorted_errors[-num_to_plot:]    # Worst performance, largest error
    plot_sets = {'Worst': worst_typhoons, 'Best': best_typhoons}

    use_cartopy = CARTOPY_AVAILABLE and all_data is not None and all_lengths is not None and id_to_indices is not None
    if CARTOPY_AVAILABLE and not use_cartopy:
        print("\nWarning: Missing full path data required for map plotting. Falling back to regular plot.")
    elif not CARTOPY_AVAILABLE:
        print("\nWarning: Cartopy library not installed. Falling back to regular plot.")

    for category, typhoons_to_plot in plot_sets.items():
        print(f"\nPlotting {len(typhoons_to_plot)} typhoons for category: {category}...")
        for i, summary in enumerate(typhoons_to_plot):
            typhoon_id_str = summary['id']
            results = results_by_id.get(typhoon_id_str)
            if not results:
                print(f"Warning: Could not find results for typhoon '{typhoon_id_str}'.")
                continue

            true_coords = np.array(results['true'])
            pred_coords = np.array(results['pred'])

            # --- Plotting ---
            if use_cartopy:
                fig = plt.figure(figsize=(12, 10))
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.add_feature(cfeature.OCEAN)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')

                # Overlay full paths (if available)
                try:
                    # id_to_indices maps int id -> [indices], here id is string, need to map
                    pass
                except Exception:
                    pass
                # Connect path: True (blue solid line) and predicted (red dashed line)
                ax.plot(true_coords[:, 1], true_coords[:, 0], 'b-', label='True Path', linewidth=1.5, transform=ccrs.Geodetic(), zorder=2)
                ax.plot(pred_coords[:, 1], pred_coords[:, 0], 'r--', label='Predicted Path', linewidth=1.2, transform=ccrs.Geodetic(), zorder=2)
                # Scatter points: True and predicted
                ax.scatter(true_coords[:, 1], true_coords[:, 0], c='b', s=18, marker='o', label='True Points', transform=ccrs.Geodetic(), zorder=3)
                ax.scatter(pred_coords[:, 1], pred_coords[:, 0], c='r', s=24, marker='x', label='Pred Points', transform=ccrs.Geodetic(), zorder=3)
                # Mark start points
                ax.plot(true_coords[0, 1], true_coords[0, 0], 'g^', markersize=8, transform=ccrs.Geodetic(), label='True Start', zorder=4)
                ax.plot(pred_coords[0, 1], pred_coords[0, 0], 'ms', markersize=8, transform=ccrs.Geodetic(), label='Pred Start', zorder=4)

                # Auto-adjust range
                buffer = 3
                all_lons = np.concatenate([true_coords[:, 1], pred_coords[:, 1]])
                all_lats = np.concatenate([true_coords[:, 0], pred_coords[:, 0]])
                ax.set_extent([all_lons.min() - buffer, all_lons.max() + buffer,
                               all_lats.min() - buffer, all_lats.max() + buffer], crs=ccrs.PlateCarree())

                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                try:
                    gl.top_labels = False
                    gl.right_labels = False
                except Exception:
                    pass
                ax.legend(loc='best', framealpha=0.8, ncol=2)
                # Use suptitle and reserve space to avoid title and plot content overlap
                fig.suptitle(
                    f"{category} Top {i+1}: Typhoon {typhoon_id_str}  |  Avg Dist Error: {summary['avg_dist_error']:.2f} km",
                    y=0.98,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(true_coords[:, 1], true_coords[:, 0], 'b-', label='True Path', linewidth=1.5, zorder=2)
                ax.plot(pred_coords[:, 1], pred_coords[:, 0], 'r--', label='Predicted Path', linewidth=1.2, zorder=2)
                # Scatter points
                ax.scatter(true_coords[:, 1], true_coords[:, 0], c='b', s=18, marker='o', label='True Points', zorder=3)
                ax.scatter(pred_coords[:, 1], pred_coords[:, 0], c='r', s=24, marker='x', label='Pred Points', zorder=3)
                # Start points
                ax.plot(true_coords[0, 1], true_coords[0, 0], 'g^', markersize=8, label='True Start', zorder=4)
                ax.plot(pred_coords[0, 1], pred_coords[0, 0], 'ms', markersize=8, label='Pred Start', zorder=4)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_aspect('equal', adjustable='datalim')
                ax.legend(loc='best', framealpha=0.8, ncol=2)
                fig.suptitle(
                    f"{category} Top {i+1}: Typhoon {typhoon_id_str}  |  Avg Dist Error: {summary['avg_dist_error']:.2f} km",
                    y=0.98,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

            filename_prefix = "worst" if category == 'Worst' else "best"
            save_path = os.path.join(plots_save_dir, f"{filename_prefix}_{i+1}_{typhoon_id_str}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  - Saved '{save_path}'")


# Build results and errors based on test set IDs, true coordinates, and predicted coordinates:
def build_results_by_id_and_errors(ids_list, true_coords, pred_coords, int_to_id_map):
    """
    Build results and errors based on test set IDs, true coordinates, and predicted coordinates:
      - results_by_id: { id_str: {'true': [...], 'pred': [...] } }
      - sorted_errors: List sorted by average distance error [{ 'id': id_str, 'avg_dist_error': float }, ...]
    """
    from collections import defaultdict
    results_by_id = defaultdict(lambda: {'true': [], 'pred': []})
    per_id_errors = defaultdict(list)

    for idx in range(len(ids_list)):
        id_int = int(ids_list[idx])
        id_str = int_to_id_map.get(id_int, str(id_int)) if isinstance(int_to_id_map, dict) else str(id_int)
        t = true_coords[idx]
        p = pred_coords[idx]
        results_by_id[id_str]['true'].append(t)
        results_by_id[id_str]['pred'].append(p)
        per_id_errors[id_str].append(haversine_distance(t[0], t[1], p[0], p[1]))

    sorted_errors = []
    for k, v in per_id_errors.items():
        sorted_errors.append({'id': k, 'avg_dist_error': float(np.mean(v))})
    sorted_errors.sort(key=lambda x: x['avg_dist_error'])
    return results_by_id, sorted_errors


# ============================================================
# Data Processing
# ============================================================

# Load Data
def load_data(filepath):
    """Load typhoon dataset."""
    with np.load(filepath, allow_pickle=True) as data:
        return data['data'], data['ids'], data['lengths'], data['feature_names']

# Create Sliding Window Samples
def create_sliding_window_samples(data, ids, lengths, window_size, prediction_horizon):
    """
    Create sliding window samples from typhoon data and track each sample's corresponding typhoon ID.
    """
    X, y, sample_ids = [], [], []
    skipped_typhoons = []
    # The first two columns are our targets: longitude and latitude.
    target_data = data[:, :, :2]

    for i, typhoon_id in enumerate(ids):
        length = lengths[i]
        
        # Check if the typhoon length is sufficient to create at least one sample
        min_length_required = window_size + prediction_horizon
        if length < min_length_required:
            skipped_typhoons.append(typhoon_id)
            continue # Skip this typhoon

        # Effective length for creating samples
        effective_len = length - min_length_required + 1
        for j in range(effective_len):
            window_end = j + window_size
            prediction_end = window_end + prediction_horizon

            X.append(data[i, j:window_end, :])
            y.append(target_data[i, prediction_end-1, :])
            sample_ids.append(typhoon_id)

    if skipped_typhoons:
        print(f"\nNote: Skipped {len(skipped_typhoons)} typhoons because their length is less than {min_length_required} (window_size + prediction_horizon).")
    
    return np.array(X), np.array(y), np.array(sample_ids)

# Define feature scaling function
def scale_features_per_column(data, scalers_list):
    scaled_data = np.copy(data)
    for i in range(data.shape[2]):
        feature_col = data[:, :, i].reshape(-1, 1)
        non_nan_mask = ~np.isnan(feature_col).flatten()
        if np.any(non_nan_mask):
            feature_col_valid = feature_col[non_nan_mask].reshape(-1, 1)
            scaled_feature_col = scalers_list[i].transform(feature_col_valid)
            original_indices = np.where(non_nan_mask)[0]
            scaled_data_flat = scaled_data[:, :, i].flatten()
            scaled_data_flat[original_indices] = scaled_feature_col.flatten()
            scaled_data[:, :, i] = scaled_data_flat.reshape(data.shape[0], data.shape[1])
    return scaled_data


# ============================================================
# Models
# ============================================================
# ConvLSTM-GAN Model
class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM unit.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM model, can contain multiple ConvLSTMCell layers.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.batch_first:
            layer_output_list = [layer_output.permute(1, 0, 2, 3, 4) for layer_output in layer_output_list]

        return layer_output_list[-1], last_state_list[-1]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class Generator(nn.Module):
    """
    Generator using ConvLSTM as the backbone.
    """
    def __init__(self, input_features, noise_dim, convlstm_hidden_dim, convlstm_kernel_size, 
                 convlstm_layers, output_size):
        super(Generator, self).__init__()
        self.input_features = input_features
        self.noise_dim = noise_dim
        
        # Reshape input features into a HxW "image" to support (3,3) convolution
        # Constraint: INPUT_FEATURES must be divisible by FEATURE_GRID_H
        self.h = FEATURE_GRID_H
        if self.input_features % self.h != 0:
            raise ValueError(
                f"INPUT_FEATURES={self.input_features} cannot be divided by FEATURE_GRID_H={self.h}; "
                "please adjust FEATURE_GRID_H or pad/crop input features so that H*W=INPUT_FEATURES."
            )
        self.w = self.input_features // self.h
        
        self.conv_lstm = ConvLSTM(input_dim=1,  # Number of input channels
                                  hidden_dim=convlstm_hidden_dim,
                                  kernel_size=convlstm_kernel_size,
                                  num_layers=convlstm_layers,
                                  batch_first=True,
                                  bias=True)
        
        # FC layer input dim = ConvLSTM output dim + noise dim
        fc_input_dim = convlstm_hidden_dim * self.h * self.w + noise_dim
        self.fc = nn.Linear(fc_input_dim, output_size)

    def forward(self, x, noise):
        # x: (B, T, F)
        b, t, f = x.shape
        
        # Reshape to ConvLSTM input format: (B, T, C, H, W)
        x_conv = x.view(b, t, 1, self.h, self.w)
        
        # ConvLSTM forward propagation
        layer_output, _ = self.conv_lstm(x_conv)
        
        # Get output from the last time step
        last_time_step_output = layer_output[:, -1, :, :, :] # (B, hidden_dim, H, W)
        
        # Flatten ConvLSTM output
        conv_out_flat = last_time_step_output.reshape(b, -1)
        
        # Concatenate noise
        combined = torch.cat([conv_out_flat, noise], dim=1)
        
        # Pass through fully connected layer
        out = self.fc(combined)
        
        return out

class Discriminator(nn.Module):
    """Social GAN Discriminator"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        
        # Discriminator only processes longitude and latitude features
        self.encoder = nn.LSTM(
            2,  # Only use lat/lon features
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        return self.classifier(h_n[-1])

# ============================================================
# Training & DDP Setup
# ============================================================

# Initialize distributed process group
def setup(rank, world_size, sync_file_path):
    """Initialize distributed process group."""
    try:
        if world_size == 1:
            # Single GPU case, use simple TCPStore, set timeout
            store = dist.TCPStore('127.0.0.1', 29500, world_size, rank == 0, timeout=timedelta(seconds=30))
        else:
            # Multi GPU case, use FileStore
            store = dist.FileStore(sync_file_path, world_size)
        dist.init_process_group(backend='gloo', store=store, rank=rank, world_size=world_size)
        print(f"[Rank {rank}] DDP initialization successful!")
    except Exception as e:
        print(f"[Rank {rank}] DDP initialization failed: {e}")
        raise

# Cleanup distributed process group
def cleanup():
    """Cleanup distributed process group."""
    try:
        dist.destroy_process_group()
    except:
        pass

# Single GPU training function (without DDP)
def train_single_gpu(train_dataset, val_dataset, test_dataset, scalers, id_to_int_map, int_to_id_map, all_data, all_lengths, id_to_indices):
    """Single GPU training function (without DDP)"""
    
    # --- Redirect standard output to log file ---
    logger = Logger()  # Use current time to generate log file name
    sys.stdout = logger
    print(f"--- Starting training on single GPU. Output will be logged to {logger.log.name} ---")

    # --- Use GPU or CPU ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Create data loaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Initialize model ---
    generator = Generator(
        input_features=INPUT_FEATURES,
        noise_dim=NOISE_DIM,
        convlstm_hidden_dim=CONVLSTM_HIDDEN_DIM,
        convlstm_kernel_size=CONVLSTM_KERNEL_SIZE,
        convlstm_layers=CONVLSTM_LAYERS,
        output_size=OUTPUT_FEATURES
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=2, hidden_dim=D_HIDDEN_DIM, num_layers=2
    ).to(device)

    # --- Define loss function and optimizer ---
    mse_criterion = nn.MSELoss().to(device)
    bce_criterion = nn.BCELoss().to(device)
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # --- Train model ---
    print("\nStarting training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        g_losses, d_losses, train_l2_losses = [], [], []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Train discriminator
            d_optimizer.zero_grad()
            input_coords = inputs[:, :, :2]
            real_traj = torch.cat([input_coords, labels.unsqueeze(1)], dim=1)
            real_score = discriminator(real_traj)
            
            noise = torch.randn(batch_size, NOISE_DIM).to(device)
            with torch.no_grad():
                fake_pred = generator(inputs, noise)
            fake_traj = torch.cat([input_coords, fake_pred.unsqueeze(1)], dim=1)
            fake_score = discriminator(fake_traj)
            
            d_loss = -(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8)).mean()
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            fake_pred = generator(inputs, noise)
            fake_traj = torch.cat([input_coords, fake_pred.unsqueeze(1)], dim=1)
            fake_score = discriminator(fake_traj)
            
            g_loss_gan = -torch.log(fake_score + 1e-8).mean()
            g_loss_l2 = mse_criterion(fake_pred, labels)
            g_loss = LAMBDA_GAN * g_loss_gan + LAMBDA_L2 * g_loss_l2
            
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            train_l2_losses.append(g_loss_l2.item())

        # --- Validate ---
        generator.eval()
        val_dist_errors, val_losses = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                noise = torch.randn(inputs.size(0), NOISE_DIM).to(device)
                pred = generator(inputs, noise)
                
                val_losses.append(mse_criterion(pred, labels).item())
                
                true_coords = np.hstack([scalers[0].inverse_transform(labels.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(labels.cpu().numpy()[:, 1:2])])
                pred_coords = np.hstack([scalers[0].inverse_transform(pred.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(pred.cpu().numpy()[:, 1:2])])
                
                for i in range(len(true_coords)):
                    val_dist_errors.append(haversine_distance(true_coords[i, 0], true_coords[i, 1], pred_coords[i, 0], pred_coords[i, 1]))

        avg_val_loss = np.mean(val_losses)
        avg_val_distance_error = np.mean(val_dist_errors)

        print(f"Epoch [{epoch+1}/{EPOCHS}]:")
        print(f"  G Loss: {np.mean(g_losses):.4f}, D Loss: {np.mean(d_losses):.4f}")
        print(f"  Train L2 Loss: {np.mean(train_l2_losses):.6f}")
        print(f"  Validation Loss: {avg_val_loss:.6f}, Validation Distance Error: {avg_val_distance_error:.2f} km")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(generator.state_dict(), MODEL_SAVE_PATH)
            print(f"    -> Validation loss improved, saving model to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"    -> Validation loss did not improve ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered!")
                break
    
    print("\nTraining finished.")
    
    # --- Evaluate final model on test set ---
    print("\nEvaluating final model on test set...")
    generator.load_state_dict(torch.load(MODEL_SAVE_PATH))
    generator.eval()
    
    test_losses, true_coords_test, pred_coords_test, ids_test_eval = [], [], [], []
    with torch.no_grad():
        for inputs, labels, ids in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            noise = torch.randn(inputs.size(0), NOISE_DIM).to(device)
            pred = generator(inputs, noise)
            test_losses.append(mse_criterion(pred, labels).item())
            
            true_coords_test.extend(np.hstack([scalers[0].inverse_transform(labels.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(labels.cpu().numpy()[:, 1:2])]))
            pred_coords_test.extend(np.hstack([scalers[0].inverse_transform(pred.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(pred.cpu().numpy()[:, 1:2])]))
            ids_test_eval.extend(ids.numpy())
    
    avg_test_loss = np.mean(test_losses)
    true_coords_test = np.array(true_coords_test)
    pred_coords_test = np.array(pred_coords_test)
    
    test_distances = [haversine_distance(t[0], t[1], p[0], p[1]) for t, p in zip(true_coords_test, pred_coords_test)]
    avg_test_distance_error = np.mean(test_distances)
    print(f"Final Test Set Loss: {avg_test_loss:.6f}, Final Test Set Average Distance Error: {avg_test_distance_error:.2f} km")

    # --- Build results and plot ---
    try:
        results_by_id, sorted_errors = build_results_by_id_and_errors(ids_test_eval, true_coords_test, pred_coords_test, int_to_id_map)
        # Plot the top and bottom typhoons
        plot_top_bottom_typhoons(sorted_errors, results_by_id, num_to_plot=10,
                                 all_data=all_data, all_lengths=all_lengths, id_to_indices=id_to_indices)
    except Exception as e:
        print(f"Plotting skipped due to error: {e}")


def main_worker(rank, world_size, sync_file_path, args):
    """Each GPU running function."""
    setup(rank, world_size, sync_file_path)
    
    # Only main process prints
    is_main_process = rank == 0

    # --- Redirect standard output to log file (only main process) ---
    if is_main_process:
        logger = Logger()  # Use current time to generate log file name
        sys.stdout = logger
        print(f"--- Starting new training run on {world_size} GPUs. Output will be logged to {logger.log.name} ---")

    # --- Unpack parameters ---
    train_dataset, val_dataset, test_dataset, scalers, id_to_int_map, int_to_id_map, all_data, all_lengths, id_to_indices = args

    # --- 1. Create distributed data loaders ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # Each GPU's batch size
    per_gpu_batch_size = BATCH_SIZE 
    
    train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
    # Validation and test typically done on single GPU or each GPU independently, here we do on each GPU
    val_loader = DataLoader(val_dataset, batch_size=per_gpu_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=per_gpu_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 2. Initialize model and move to specified GPU ---
    # Each process corresponds to one GPU, use explicit torch.device
    device = torch.device(f'cuda:{rank}')
    
    generator = Generator(
        input_features=INPUT_FEATURES,
        noise_dim=NOISE_DIM,
        convlstm_hidden_dim=CONVLSTM_HIDDEN_DIM,
        convlstm_kernel_size=CONVLSTM_KERNEL_SIZE,
        convlstm_layers=CONVLSTM_LAYERS,
        output_size=OUTPUT_FEATURES
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=2, hidden_dim=D_HIDDEN_DIM, num_layers=2 # Discriminator can remain simple
    ).to(device)

    # Use DDP to wrap the model
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    if is_main_process:
        print("Using DistributedDataParallel (DDP) for training.")
        print(f"  - Total Batch Size: {per_gpu_batch_size * world_size}")

    # --- 3. Define loss function and optimizer ---
    mse_criterion = nn.MSELoss().to(device)
    bce_criterion = nn.BCELoss().to(device)
    
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # --- 4. Train model ---
    if is_main_process:
        print("\nStarting training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Set epoch, to ensure each epoch has different shuffling
        train_loader.sampler.set_epoch(epoch)
        
        generator.train()
        discriminator.train()
        
        g_losses, d_losses, train_l2_losses = [], [], []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Train discriminator
            d_optimizer.zero_grad()
            input_coords = inputs[:, :, :2]
            real_traj = torch.cat([input_coords, labels.unsqueeze(1)], dim=1)
            real_score = discriminator(real_traj)

            noise = torch.randn(batch_size, NOISE_DIM).to(device)
            with torch.no_grad():
                fake_pred = generator(inputs, noise)
            fake_traj = torch.cat([input_coords, fake_pred.unsqueeze(1)], dim=1)
            fake_score = discriminator(fake_traj)

            d_loss = -(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8)).mean()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            fake_pred = generator(inputs, noise)
            fake_traj = torch.cat([input_coords, fake_pred.unsqueeze(1)], dim=1)
            fake_score = discriminator(fake_traj)

            g_loss_gan = -torch.log(fake_score + 1e-8).mean()
            g_loss_l2 = mse_criterion(fake_pred, labels)
            g_loss = LAMBDA_GAN * g_loss_gan + LAMBDA_L2 * g_loss_l2

            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            train_l2_losses.append(g_loss_l2.item())

        # --- Validate ---
        generator.eval()
        val_dist_errors, val_losses = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                noise = torch.randn(inputs.size(0), NOISE_DIM).to(device)
                pred = generator(inputs, noise)

                val_losses.append(mse_criterion(pred, labels).item())

                true_coords = np.hstack([scalers[0].inverse_transform(labels.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(labels.cpu().numpy()[:, 1:2])])
                pred_coords = np.hstack([scalers[0].inverse_transform(pred.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(pred.cpu().numpy()[:, 1:2])])

                for i in range(len(true_coords)):
                    val_dist_errors.append(haversine_distance(true_coords[i, 0], true_coords[i, 1], pred_coords[i, 0], pred_coords[i, 1]))

        avg_val_loss = np.mean(val_losses)
        avg_val_distance_error = np.mean(val_dist_errors)

        # Sync validation loss across all processes
        val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss_synced = val_loss_tensor.item() / world_size

        if is_main_process:
            print(f"Epoch [{epoch+1}/{EPOCHS}]:")
            print(f"  G Loss: {np.mean(g_losses):.4f}, D Loss: {np.mean(d_losses):.4f}")
            print(f"  Train L2 Loss: {np.mean(train_l2_losses):.6f}")
            print(f"  Validation Loss: {avg_val_loss_synced:.6f}, Validation Distance Error: {avg_val_distance_error:.2f} km")

            # Early stopping and model saving
            if avg_val_loss_synced < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss_synced
                patience_counter = 0
                # Save model when, to save .module attributes
                torch.save(generator.module.state_dict(), MODEL_SAVE_PATH)
                print(f"    -> Validation loss improved, saving model to {MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(f"    -> Validation loss did not improve ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered!")
                    break
    
    if is_main_process:
        print("\nTraining finished.")
        # --- 5. Evaluate final model on test set (only on main process) ---
        print("\nEvaluating final model on test set...")
        # Load best model
        generator.module.load_state_dict(torch.load(MODEL_SAVE_PATH))
        generator.eval()
        
        test_losses, true_coords_test, pred_coords_test, ids_test_eval = [], [], [], []
        with torch.no_grad():
            for inputs, labels, ids in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                noise = torch.randn(inputs.size(0), NOISE_DIM).to(device)
                pred = generator(inputs, noise)
                test_losses.append(mse_criterion(pred, labels).item())
                
                true_coords_test.extend(np.hstack([scalers[0].inverse_transform(labels.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(labels.cpu().numpy()[:, 1:2])]))
                pred_coords_test.extend(np.hstack([scalers[0].inverse_transform(pred.cpu().numpy()[:, 0:1]), scalers[1].inverse_transform(pred.cpu().numpy()[:, 1:2])]))
                ids_test_eval.extend(ids.numpy())
        
        # ... (subsequent detailed evaluation report logic remains unchanged) ...
        # (for brevity, omitted)
        avg_test_loss = np.mean(test_losses)
        true_coords_test = np.array(true_coords_test)
        pred_coords_test = np.array(pred_coords_test)
        
        test_distances = [haversine_distance(t[0], t[1], p[0], p[1]) for t, p in zip(true_coords_test, pred_coords_test)]
        avg_test_distance_error = np.mean(test_distances)
        print(f"Final Test Set Loss: {avg_test_loss:.6f}, Final Test Set Average Distance Error: {avg_test_distance_error:.2f} km")

        # --- Build results and plot (only main process) ---
        try:
            results_by_id, sorted_errors = build_results_by_id_and_errors(ids_test_eval, true_coords_test, pred_coords_test, int_to_id_map)
            plot_top_bottom_typhoons(sorted_errors, results_by_id, num_to_plot=10,
                                     all_data=all_data, all_lengths=all_lengths, id_to_indices=id_to_indices)
        except Exception as e:
            print(f"Plotting skipped due to error: {e}")


    cleanup()

# ============================================================
# Main Execution Block
# ============================================================
if __name__ == '__main__':
    # Load and preprocess data
    all_data, all_ids, all_lengths, feature_names = load_data(DATA_PATH)
    unique_ids_str = np.unique(all_ids)
    id_to_int_map = {id_str: i for i, id_str in enumerate(unique_ids_str)}
    int_to_id_map = {i: id_str for id_str, i in id_to_int_map.items()}
    all_ids_int = np.array([id_to_int_map[id_str] for id_str in all_ids])
    unique_ids_int = np.unique(all_ids_int)
    train_val_ids, test_ids = train_test_split(unique_ids_int, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE)
    val_split_ratio = VALIDATION_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_split_ratio, random_state=RANDOM_STATE)
    id_to_indices = collections.defaultdict(list)
    for i, id_int in enumerate(all_ids_int):
        id_to_indices[id_int].append(i)
    train_indices = [idx for i in train_ids for idx in id_to_indices[i]]
    val_indices = [idx for i in val_ids for idx in id_to_indices[i]]
    test_indices = [idx for i in test_ids for idx in id_to_indices[i]]
    data_train, data_val, data_test = all_data[train_indices], all_data[val_indices], all_data[test_indices]
    ids_train_int, ids_val_int, ids_test_int = all_ids_int[train_indices], all_ids_int[val_indices], all_ids_int[test_indices]
    lengths_train, lengths_val, lengths_test = all_lengths[train_indices], all_lengths[val_indices], all_lengths[test_indices]
    data_train, data_val, data_test = data_train[:, :, :INPUT_FEATURES], data_val[:, :, :INPUT_FEATURES], data_test[:, :, :INPUT_FEATURES]
    scalers = [StandardScaler() for _ in range(INPUT_FEATURES)]
    for i in range(INPUT_FEATURES):
        feature_col = data_train[:, :, i].reshape(-1, 1)
        valid_feature_data = feature_col[~np.isnan(feature_col)]
        if len(valid_feature_data) > 0:
            scalers[i].fit(valid_feature_data.reshape(-1, 1))
    


    processed_data_train = scale_features_per_column(data_train, scalers)
    processed_data_val = scale_features_per_column(data_val, scalers)
    processed_data_test = scale_features_per_column(data_test, scalers)
    
    X_train, y_train, _ = create_sliding_window_samples(processed_data_train, ids_train_int, lengths_train, WINDOW_SIZE, PREDICTION_HORIZON)
    X_val, y_val, _ = create_sliding_window_samples(processed_data_val, ids_val_int, lengths_val, WINDOW_SIZE, PREDICTION_HORIZON)
    X_test, y_test, ids_test_samples = create_sliding_window_samples(processed_data_test, ids_test_int, lengths_test, WINDOW_SIZE, PREDICTION_HORIZON)
    
    X_train_tensor, y_train_tensor = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val_tensor, y_val_tensor = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    X_test_tensor, y_test_tensor, ids_test_tensor = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(ids_test_samples).int()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, ids_test_tensor)

    # Set up and start training
    world_size = torch.cuda.device_count()
    if world_size == 1:
        # Single GPU, direct training
        print(f"Found {world_size} GPU. Running single-process training (no DDP).")
        args = (train_dataset, val_dataset, test_dataset, scalers, id_to_int_map, int_to_id_map, all_data, all_lengths, id_to_indices)
        train_single_gpu(*args)
    elif world_size > 1:
        # Multi GPU, use DDP
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        args = (train_dataset, val_dataset, test_dataset, scalers, id_to_int_map, int_to_id_map, all_data, all_lengths, id_to_indices)
        sync_file_path = "distributed_sync_file"
        mp.spawn(main_worker,
                 args=(world_size, sync_file_path, args),
                 nprocs=world_size,
                 join=True)
    else:
        print("No GPU found. Running on CPU (single-process, no DDP).")
        args = (train_dataset, val_dataset, test_dataset, scalers, id_to_int_map, int_to_id_map, all_data, all_lengths, id_to_indices)
        train_single_gpu(*args)
