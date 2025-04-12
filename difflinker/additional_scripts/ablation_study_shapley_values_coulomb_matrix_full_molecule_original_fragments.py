#!/usr/bin/env python
# coding: utf-8

# ## Abaltion study on important/opposing features
# 
# Abaltion study consisting in keeping only important atoms for the generation to inspect the model's behavior

# 1. Read molecules from dataset
# 2. Load initial random distributions for atoms and features
# 3. Read shapley values
# 4. Consider Shapley value importances according to avg value
# 5. Add or remove the most important atoms one by one
# 
# 

# ### Import Libraries

# In[1]:


import os
os.environ["http_proxy"] = "http://web-proxy.informatik.uni-bonn.de:3128"
os.environ["https_proxy"] = "http://web-proxy.informatik.uni-bonn.de:3128"

import yaml
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import networkx as nx
from pysmiles import read_smiles

# from sklearn.decomposition import PCA

import torch
from sklearn.decomposition import PCA

from src.lightning import DDPM
from src.datasets import get_dataloader
from src.visualizer import load_molecule_xyz, load_xyz_files, save_xyz_file
from src.molecule_builder import get_bond_order
from src import const


# In[2]:


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# density = sys.argv[sys.argv.index("--P") + 1]
# seed = sys.argv[sys.argv.index("--seed") + 1]

# Load configuration from config.yml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

checkpoint = config['CHECKPOINT']
chains = config['CHAINS']
DATA = config['DATA']
prefix = config['PREFIX']
keep_frames = int(config['KEEP_FRAMES'])
P = config['P']
device = config['DEVICE'] if torch.cuda.is_available() else 'cpu'
SEED = int(config['SEED'])
REMOVAL = config['REMOVAL']


# In[3]:


experiment_name = checkpoint.split('/')[-1].replace('.ckpt', '')

#create output directories
chains_output_dir = os.path.join(chains, experiment_name, prefix, 'chains_' + P + '_seed_' + str(SEED) + '_ablation_study')
final_states_output_dir = os.path.join(chains, experiment_name, prefix, 'final_states_' + P + '_seed_' + str(SEED) + '_ablation_study')
os.makedirs(chains_output_dir, exist_ok=True)
os.makedirs(final_states_output_dir, exist_ok=True)

# Loading model form checkpoint 
model = DDPM.load_from_checkpoint(checkpoint, map_location=device)

# Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
model.val_data_prefix = prefix

print(f"Running on device: {device}")
# In case <Anonymous> will run my model or vice versa
if DATA is not None:
    model.data_path = DATA

model = model.eval().to(device)
model.setup(stage='val')
dataloader = get_dataloader(
    model.val_dataset,
    batch_size=1, #@mastro, it was 32
    # batch_size=len(model.val_dataset)
)


# ### Set random seeds

# In[4]:


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)


# ### Utility functions

# In[5]:


def arrestomomentum():
    raise KeyboardInterrupt("Debug interrupt.")

def draw_sphere_xai(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) #* 0.8
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, alpha=alpha)

def plot_molecule_xai(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, is_geom, fragment_mask=None, phi_values=None, colors_fragment_shadow=None):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    colors_dic = np.array(const.COLORS)
    radius_dic = np.array(const.RADII)
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if fragment_mask is None:
        fragment_mask = torch.ones(len(x))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = idx2atom[atom_type[i]], idx2atom[atom_type[j]]
            draw_edge_int = get_bond_order(atom1, atom2, dist)
            line_width = (3 - 2) * 2 * 2
            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1
                linewidth_factor *= 0.5
                ax.plot(
                    [x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                    linewidth=line_width * linewidth_factor * 2,
                    c=hex_bg_color,
                    alpha=alpha
                )

    

    if spheres_3d:
        
        for i, j, k, s, c, f, phi in zip(x, y, z, radii, colors, fragment_mask, phi_values):
            if f == 1:
                alpha = 1.0
                if phi > 0:
                    c = 'red'

            draw_sphere_xai(ax, i.item(), j.item(), k.item(), 0.5 * s, c, alpha)

    else:

        #draw fragments
        fragment_mask_on_cpu = fragment_mask.cpu().numpy()
        colors_fragment = colors[fragment_mask_on_cpu == 1]
        x_fragment = x[fragment_mask_on_cpu == 1]
        y_fragment = y[fragment_mask_on_cpu == 1]
        z_fragment = z[fragment_mask_on_cpu == 1]
        areas_fragment = areas[fragment_mask_on_cpu == 1]
        
        if phi_values is not None and colors_fragment_shadow is None:
            phi_values_array = np.array(phi_values)
            # Calculate the gradient colors based on phi values
            cmap = plt.cm.get_cmap('coolwarm_r') #reversed heatmap for distance-based importance
            norm = plt.Normalize(vmin=min(phi_values_array), vmax=max(phi_values_array))
            colors_fragment_shadow = cmap(norm(phi_values_array))
        elif colors_fragment_shadow is not None and phi_values is None:
            colors_fragment_shadow = colors_fragment_shadow
        else:
            raise ValueError("Either phi_values or colors_fragment_shadow must be provided, not both.")
        # ax.scatter(x_fragment, y_fragment, z_fragment, s=areas_fragment, alpha=0.9 * alpha, c=colors_fragment)

        ax.scatter(x_fragment, y_fragment, z_fragment, s=areas_fragment, alpha=0.9 * alpha, c=colors_fragment, edgecolors=colors_fragment_shadow, linewidths=5, rasterized=False)

        #draw non-fragment atoms
        colors = colors[fragment_mask_on_cpu == 0]
        x = x[fragment_mask_on_cpu == 0]
        y = y[fragment_mask_on_cpu == 0]
        z = z[fragment_mask_on_cpu == 0]
        areas = areas[fragment_mask_on_cpu == 0]
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors, rasterized=False)


def plot_data3d_xai(positions, atom_type, is_geom, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1., fragment_mask=None, phi_values=None, colors_fragment_shadow=None):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#000000' #'#666666'

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule_xai(
        ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, is_geom=is_geom, fragment_mask=fragment_mask, phi_values=phi_values, colors_fragment_shadow=colors_fragment_shadow
    )

    max_value = positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    dpi = 300 if spheres_3d else 300 #it was 120 and 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()

def visualize_chain_xai(
        path, spheres_3d=False, bg="black", alpha=1.0, wandb=None, mode="chain", is_geom=False, fragment_mask=None, phi_values=None, colors_fragment_shadow=None
):
    files = load_xyz_files(path)
    save_paths = []

    # Fit PCA to the final molecule – to obtain the best orientation for visualization
    positions, one_hot, charges = load_molecule_xyz(files[-1], is_geom=is_geom)
    pca = PCA(n_components=3)
    pca.fit(positions)

    for i in range(len(files)):
        file = files[i]

        positions, one_hot, charges = load_molecule_xyz(file, is_geom=is_geom)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        # Transform positions of each frame according to the best orientation of the last frame
        positions = pca.transform(positions)
        positions = torch.tensor(positions)

        fn = file[:-4] + '.png'
        plot_data3d_xai(
            positions, atom_type,
            save_path=fn,
            spheres_3d=spheres_3d,
            alpha=alpha,
            bg=bg,
            camera_elev=90,
            camera_azim=90,
            is_geom=is_geom,
            fragment_mask=fragment_mask,
            phi_values=phi_values,
            colors_fragment_shadow=colors_fragment_shadow
        )
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


# ### Generation and Ablation Study

# In[6]:


num_samples = 30
sampled = 0
start = 0

SAVE_VISUALIZATION = config['SAVE_VISUALIZATION']
SAVE_EXPLANATION_PATH = "results/explanations_coulomb_" + P + "_seed_" + str(SEED)

data_list = []
for data in dataloader:

    if sampled < num_samples:
        data_list.append(data)
        sampled += 1

max_num_atoms = max(data["positions"].shape[1] for data in data_list)

# load initial distrubution of noisy features and positions
INTIAL_DISTIBUTION_PATH = "results/explanations_" + P + "_seed_" + str(SEED)

noisy_features = torch.load(INTIAL_DISTIBUTION_PATH + "/noisy_features_seed_" + str(SEED) + ".pt", map_location=device, weights_only=True)
noisy_positions = torch.load(INTIAL_DISTIBUTION_PATH + "/noisy_positions_seed_" + str(SEED) + ".pt", map_location=device, weights_only=True)


# ### Incremental generation by adding atoms according to thier Shapley values (for the more to the less important)

# In[7]:


#create output directories
# chains_output_dir = os.path.join(chains, experiment_name, prefix, 'chains_' + P + '_seed_' + str(SEED) + '_ablation_study_minimal_sets')
# final_states_output_dir = os.path.join(chains, experiment_name, prefix, 'final_states_' + P + '_seed_' + str(SEED) + '_ablation_study_minimal_sets')
chains_output_dir = None
final_states_output_dir = None

if REMOVAL:
    chains_output_dir = os.path.join(chains, experiment_name, prefix, 'chains_' + P + '_seed_' + str(SEED) + '_ablation_study_coulomb_matrix_minimal_sets_atom_removal')
    final_states_output_dir = os.path.join(chains, experiment_name, prefix, 'final_states_' + P + '_seed_' + str(SEED) + '_ablation_study_coulomb_matrix_minimal_sets_atom_removal')
else:
    chains_output_dir = os.path.join(chains, experiment_name, prefix, 'chains_' + P + '_seed_' + str(SEED) + '_ablation_study_coulomb_matrix_minimal_sets_atom_addition')
    final_states_output_dir = os.path.join(chains, experiment_name, prefix, 'final_states_' + P + '_seed_' + str(SEED) + '_ablation_study_coulomb_matrix_minimal_sets_atom_addition')
    
os.makedirs(chains_output_dir, exist_ok=True)
os.makedirs(final_states_output_dir, exist_ok=True)

for data_index, data in enumerate(tqdm(data_list)):

    smile = data["name"][0]
    mol = read_smiles(smile)
    chain_with_full_fragments = None

    noisy_positions_present_atoms = noisy_positions.clone()
    noisy_features_present_atoms = noisy_features.clone()

    noisy_positions_present_atoms = noisy_positions_present_atoms[:, :data["positions"].shape[1], :]
    noisy_features_present_atoms = noisy_features_present_atoms[:, :data["one_hot"].shape[1], :]

    num_fragment_atoms = int(data["fragment_mask"].sum().item())

    #load Shapley values 
    phi_values = []
    
    
    with open(SAVE_EXPLANATION_PATH + "/phi_atoms_" + str(data_index) + ".txt", "r") as read_file:
        read_file.readline()
        read_file.readline()
        for row in read_file:
            if row.strip() == "":
                break
            line = row.strip().split(",")
            phi_values.append(float(line[1])) #1 for Frobenius norm Shapley values from coulomb matrix

    #retrieve original and average Frobenius norm from file
    original_frobenius_norm = None
    average_frobenius_norm = None
    with open(SAVE_EXPLANATION_PATH + "/phi_atoms_" + str(data_index) + ".txt", "r") as read_file:
        for row in read_file:
            if row.strip().startswith("Frobenius norm of original molecule"):
                
                line = row.strip().split(":")
                print("line", line)
                # line = ''.join(filter(lambda x: x.isdigit() or x == '.', line))
                original_frobenius_norm = float(line[1])
            if row.strip().startswith("Average Frobenius norm of random samples"):
                
                line = row.strip().split(":")
                #take only numbers and dots from line
                # line = ''.join(filter(lambda x: x.isdigit() or x == '.', line))           
                average_frobenius_norm = float(line[1])    
            if average_frobenius_norm is not None and original_frobenius_norm is not None:
                break

    print("original_frobenius_norm", original_frobenius_norm)
    print("average_frobenius_norm", average_frobenius_norm)
    # Remove fragment atoms whose Shapley values are above the average Shapley value
    fragment_mask = data["fragment_mask"].squeeze().bool()
    linker_mask = data["linker_mask"].squeeze().bool()
    phi_values_tensor = torch.tensor(phi_values)

    #get indices of phi_values_tensor from lower to higher if average_frobenius_norm > original_frobenius_norm and vice versa
    sorted_indices = None
    if average_frobenius_norm > original_frobenius_norm:
        sorted_indices = torch.argsort(phi_values_tensor)
    else:
        sorted_indices = torch.argsort(phi_values_tensor, descending=True)
    # reversed_indices = torch.flip(sorted_indices, [0])

    
    
    for sorted_index in tqdm(range(len(sorted_indices))):
        data_temp = data.copy()
        noisy_positions_present_atoms_temp = noisy_positions_present_atoms.clone()
        noisy_features_present_atoms_temp = noisy_features_present_atoms.clone()
        
        #keep indices from 0 to sorted_index
        # shapley_value_indices_keep = sorted_indices[:sorted_index+1] #for addition
        shapley_value_indices_keep = None
        if REMOVAL:
            shapley_value_indices_keep = sorted_indices[sorted_index:] #for removal
        else:
            shapley_value_indices_keep = sorted_indices[:sorted_index+1] #for addition
        # print("fragment_atoms_indices_keep", fragment_atoms_indices_keep)
        # print("len fragment_atoms_indices_keep", len(fragment_atoms_indices_keep))
        # print("phi_values_tensor", phi_values_tensor)
        
        #retrieve indices of fragment and linker atoms from atom_mask
        fragment_atoms_indices = torch.where(fragment_mask)[0]
        fragment_atoms_indices = fragment_atoms_indices.to(device)
        linker_atoms_indices = torch.where(linker_mask)[0]
        linker_atoms_indices = linker_atoms_indices.to(device)
        
        #keep only elements from fragment_atoms_indices at the indices in shapley_value_indices_keep
        fragment_atoms_indices_keep = fragment_atoms_indices[shapley_value_indices_keep]
        fragment_atoms_indices_keep_tensor = torch.Tensor(fragment_atoms_indices_keep).to(device)

        #retrieve indices of fragment atoms with Shapley values above the average Shapley value
        # fragment_atoms_to_remove_indices = torch.where(phi_values_tensor > average_phi_value)[0]
        # fragment_atoms_to_remove_indices = fragment_atoms_to_remove_indices.to(device)

        # fragment_atoms_indices_keep = torch.tensor([i for i in fragment_atoms_indices if i not in fragment_atoms_to_remove_indices])
        # fragment_atoms_indices_keep = fragment_atoms_indices_keep.to(device)

        #keep only fragment_atoms_indices_keep and linker_atoms_indices
        atom_indices_to_keep = torch.cat((fragment_atoms_indices_keep_tensor, linker_atoms_indices)).to(device)

        #remove atoms from molecule
        data_temp["positions"] = data_temp["positions"][:, atom_indices_to_keep, :]
        data_temp["one_hot"] = data_temp["one_hot"][:, atom_indices_to_keep, :]
        data_temp["charges"] = data_temp["charges"][:, atom_indices_to_keep]
        data_temp["fragment_mask"] = data_temp["fragment_mask"][:, atom_indices_to_keep]
        data_temp["linker_mask"] = data_temp["linker_mask"][:, atom_indices_to_keep]
        data_temp["atom_mask"] = data_temp["atom_mask"][:, atom_indices_to_keep]
        data_temp["anchors"] = data_temp["anchors"][:, atom_indices_to_keep]
        edge_mask_to_keep = (data_temp["atom_mask"].unsqueeze(1) * data_temp["atom_mask"]).flatten()
        data_temp["edge_mask"] = edge_mask_to_keep

        #remove atoms from noisy features and positions
        noisy_positions_present_atoms_temp = noisy_positions_present_atoms_temp[:, atom_indices_to_keep, :]
        noisy_features_present_atoms_temp = noisy_features_present_atoms_temp[:, atom_indices_to_keep, :]

        cmap = None
        if average_frobenius_norm > original_frobenius_norm:
            cmap = plt.cm.get_cmap('coolwarm_r')
        else:
            cmap = plt.cm.get_cmap('coolwarm')
        #remove atom from phi_values
        phi_values_array = np.array(phi_values)
         #reversed heatmap for distance-based importance
        norm = plt.Normalize(vmin=min(phi_values_array), vmax=max(phi_values_array))
        colors_fragment_shadow = cmap(norm(phi_values_array))
        #remove atoms from color array
        colors_fragment_shadow = colors_fragment_shadow[fragment_atoms_indices_keep.cpu().numpy()]
        
        chain_batch, node_mask = model.sample_chain(data_temp, keep_frames=keep_frames, noisy_positions=noisy_positions_present_atoms_temp, noisy_features=noisy_features_present_atoms_temp)

        # chain_with_full_fragments = chain_batch[0, :, :, :]

        #save and visualize chain (only for the linker use noisy positions for the initial distribution)
        

        for i in range(len(data_temp['positions'])):
            chain = chain_batch[:, i, :, :]
            assert chain.shape[0] == keep_frames
            assert chain.shape[1] == data_temp['positions'].shape[1]
            assert chain.shape[2] == data_temp['positions'].shape[2] + data_temp['one_hot'].shape[2] + model.include_charges

            # Saving chains
            name = str(i + start)
            chain_output = os.path.join(chains_output_dir, name, name + "_atoms_" + str(len(fragment_atoms_indices_keep)))
            os.makedirs(chain_output, exist_ok=True)
            
            #save initial random distrubution with noise
            positions_combined = torch.zeros_like(data_temp['positions'])
            one_hot_combined = torch.zeros_like(data_temp['one_hot'])

            # Iterate over each atom and decide whether to use original or noisy data
            for atom_idx in range(data_temp['positions'].shape[1]):
                if data_temp['fragment_mask'][0, atom_idx] == 1:
                    # Use original positions and features for fragment atoms
                    positions_combined[:, atom_idx, :] = data_temp['positions'][:, atom_idx, :]
                    one_hot_combined[:, atom_idx, :] = data_temp['one_hot'][:, atom_idx, :]
                    # atom_mask_combined[:, atom_idx] = data_temp['atom_mask'][:, atom_idx]
                else:
                    # Use noisy positions and features for linker atoms
                    positions_combined[:, atom_idx, :] = noisy_positions_present_atoms_temp[:, atom_idx, :]
                    one_hot_combined[:, atom_idx, :] = noisy_features_present_atoms_temp[:, atom_idx, :]

            #save initial distribution TODO: fix positions, they are not centered
            save_xyz_file(
                chain_output,
                one_hot_combined,
                positions_combined,
                node_mask[i].unsqueeze(0),
                names=[f'{name}_' + str(keep_frames)],
                is_geom=model.is_geom
            )

            # one_hot = chain[:, :, 3:-1]
            one_hot = chain[:, :, 3:] #@mastro, added last atom type (not sure whyt it was not included...) However, TODO check again
            positions = chain[:, :, :3]
            chain_node_mask = torch.cat([node_mask[i].unsqueeze(0) for _ in range(keep_frames)], dim=0)
            names = [f'{name}_{j}' for j in range(keep_frames)]

            save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=model.is_geom)

            invert_colormap = False
            if average_frobenius_norm > original_frobenius_norm:
                invert_colormap = True

            visualize_chain_xai(
                chain_output,
                spheres_3d=False,
                alpha=0.7,
                bg='white',
                is_geom=model.is_geom,
                fragment_mask=data_temp['fragment_mask'][i].squeeze(),
                phi_values=None,
                colors_fragment_shadow=colors_fragment_shadow,
            )

            # Saving final prediction and ground truth separately
            true_one_hot = data_temp['one_hot'][i].unsqueeze(0)
            true_positions = data_temp['positions'][i].unsqueeze(0)
            true_node_mask = data_temp['atom_mask'][i].unsqueeze(0)

            final_states_output_dir_current = os.path.join(final_states_output_dir, name)
            save_xyz_file(
                final_states_output_dir,
                true_one_hot,
                true_positions,
                true_node_mask,
                names=[f'{name}_true_atoms_' + str(len(fragment_atoms_indices_keep))],
                is_geom=model.is_geom,
            )

            pred_one_hot = chain[0, :, 3:-1].unsqueeze(0)
            pred_positions = chain[0, :, :3].unsqueeze(0)
            pred_node_mask = chain_node_mask[0].unsqueeze(0)
            save_xyz_file(
                final_states_output_dir,
                pred_one_hot,
                pred_positions,
                pred_node_mask,
                names=[f'{name}_pred_atoms_' + str(len(fragment_atoms_indices_keep))],
                is_geom=model.is_geom
            )

        del data_temp
        del noisy_features_present_atoms_temp
        del noisy_positions_present_atoms_temp
    start += len(data['positions'])

