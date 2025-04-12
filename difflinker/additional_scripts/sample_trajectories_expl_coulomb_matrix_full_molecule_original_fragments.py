#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["http_proxy"] = "http://web-proxy.informatik.uni-bonn.de:3128"
os.environ["https_proxy"] = "http://web-proxy.informatik.uni-bonn.de:3128"


# In[2]:


import argparse
import torch

from src.datasets import get_dataloader
from src.lightning import DDPM
from src.molecule_builder import get_bond_order
from src.visualizer import save_xyz_file
from tqdm.auto import tqdm
import sys #@mastro
from src import const #@mastro
import numpy as np #@mastro
from numpy.random import default_rng
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import directed_hausdorff
import random
from sklearn.decomposition import PCA
from src.visualizer import load_molecule_xyz, load_xyz_files
import matplotlib.pyplot as plt
import imageio
from src import const
import networkx as nx
import time 
import yaml
from pysmiles import read_smiles
#get running device from const file

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Simulate command-line arguments

# density = sys.argv[sys.argv.index("--P") + 1]
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
SAVE_VISUALIZATION = config['SAVE_VISUALIZATION']
M = int(config['M'])
NUM_SAMPLES = int(config['NUM_SAMPLES'])
PARALLEL_STEPS = int(config['PARALLEL_STEPS'])
DIAGONALIZE = config['DIAGONALIZE']
ROTATE = config['ROTATE']
TRANSLATE = config['TRANSLATE']
REFLECT = config['REFLECT']
TRANSFORMATION_SEED = int(config['TRANSFORMATION_SEED'])

print("seed is: ", SEED)

experiment_name = checkpoint.split('/')[-1].replace('.ckpt', '')

transformations = []
if ROTATE:
    transformations.append("rotate")
if TRANSLATE:
    transformations.append("translate")
if REFLECT:
    transformations.append("reflect")

transformations_str = "_".join(transformations) if transformations else "no_transform"
chains_output_dir = os.path.join(chains, experiment_name, prefix, f'chains_coulomb_matrix_full_molecule_original_fragments_{P}_seed_{SEED}_{transformations_str}_transformation_seed_{TRANSFORMATION_SEED}')
final_states_output_dir = os.path.join(chains, experiment_name, prefix, f'final_states_coulomb_matrix_full_molecule_original_fragments_{P}_seed_{SEED}_{transformations_str}_transformation_seed_{TRANSFORMATION_SEED}')

if DIAGONALIZE:
    chains_output_dir = chains_output_dir + '_diagonalized'
    final_states_output_dir = final_states_output_dir + '_diagonalized'

print("Applied trasformations: ", transformations_str)
print("Seed used for random transformations: ", TRANSFORMATION_SEED)
    
os.makedirs(chains_output_dir, exist_ok=True)
os.makedirs(final_states_output_dir, exist_ok=True)

# Loading model form checkpoint (all hparams will be automatically set)
model = DDPM.load_from_checkpoint(checkpoint, map_location=device)

# Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
model.val_data_prefix = prefix

print(f"Running device: {device}")
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



# In[3]:


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)


# #### Similarity functions

# In[4]:


def compute_molecular_similarity(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on distances and atom type.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        
    Returns:
        float: The similarity between the two molecules.
    """
    # If fragmen_mask is provided, only consider the atoms in the mask
    if mask1 is not None:
        mask1 = mask1.bool()
        mol1 = mol1[mask1,:]

    if mask2 is not None:
        mask2 = mask2.bool()
        mol2 = mol2[mask2,:]

    return 1 - torch.norm(mol1 - mol2)

def compute_molecular_distance(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on distances and atom type.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        
    Returns:
        float: The similarity between the two molecules.
    """
    # If fragmen_mask is provided, only consider the atoms in the mask
    if mask1 is not None:
        mask1 = mask1.bool()
        mol1 = mol1[mask1,:]

    if mask2 is not None:
        mask2 = mask2.bool()
        mol2 = mol2[mask2,:]

    return torch.norm(mol1 - mol2).item()

def compute_molecular_distance_batch(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on distances and atom type.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        
    Returns:
        torch.Tensor: The similarity between the two molecules for each element in the batch.
    """
    # If fragment_mask is provided, only consider the atoms in the mask
    if mask1 is not None:
        mask1 = mask1.bool()
        batch_size = mol1.shape[0]
        masked_mol1 = []
        for i in range(batch_size):
            masked_mol1.append(mol1[i, mask1[i], :])

        if batch_size == 1:
            mol1 = masked_mol1[0].unsqueeze(0)
        else:    
            mol1 = torch.stack(masked_mol1)
           
    if mask2 is not None:
        mask2 = mask2.bool()
        batch_size = mol2.shape[0]
        masked_mol2 = []
        for i in range(batch_size):
            masked_mol2.append(mol2[i, mask2[i], :])
        
        if batch_size == 1:
            mol2 = masked_mol2[0].unsqueeze(0)
        else:    
            mol2 = torch.stack(masked_mol2)

    return torch.norm(mol1 - mol2, dim=(1,2))

def compute_cosine_similarity(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on distances and atom type.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        
    Returns:
        float: The similarity between the two molecules.
    """
    # If fragmen_mask is provided, only consider the atoms in the mask
    if mask1 is not None:
        mask1 = mask1.bool()
        mol1 = mol1[mask1,:]

    if mask2 is not None:
        mask2 = mask2.bool()
        mol2 = mol2[mask2,:]

    return cosine_similarity(mol1.flatten().reshape(1, -1), mol2.flatten().reshape(1, -1)).item()


def compute_cosine_similarity_batch(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on distances and atom type.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        
    Returns:
        float: The similarity between the two molecules.
    """
    # If fragmen_mask is provided, only consider the atoms in the mask
    if mask1 is not None:
        mask1 = mask1.bool()
        batch_size = mol1.shape[0]
        masked_mol1 = []
        for i in range(batch_size):
            masked_mol1.append(mol1[i, mask1[i], :])
        
        if batch_size == 1:
            mol1 = masked_mol1[0].unsqueeze(0)
        else:    
            mol1 = torch.stack(masked_mol1)
        

    if mask2 is not None:
        mask2 = mask2.bool()
        mask2 = mask2.bool()
        batch_size = mol2.shape[0]
        masked_mol2 = []
        for i in range(batch_size):
            masked_mol2.append(mol2[i, mask2[i], :])
        
        if batch_size == 1:
            mol2 = masked_mol2[0].unsqueeze(0)
        else:    
            mol2 = torch.stack(masked_mol2)

    cos_sims = []
    for i in range(mol1.shape[0]):
        cos_sims.append(cosine_similarity(mol1[i].flatten().reshape(1, -1), mol2[i].flatten().reshape(1, -1)).item())

    return cos_sims

def compute_molecular_similarity_positions(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on positions.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        
    Returns:
        float: The similarity between the two molecules.
    """
    # If fragmen_mask is provided, only consider the atoms in the mask
    positions1 = mol1[:, :3].squeeze()
    positions2 = mol2[:, :3].squeeze()

    if mask1 is not None:
        mask1 = mask1.bool()
        positions1 = positions1[mask1,:]

    if mask2 is not None:
        mask2 = mask2.bool()
        positions2 = positions2[mask2,:]


    return 1 - torch.norm(positions1 - positions2) #choose if distance or similarity, need to check what it the better choice

def compute_one_hot_similarity(mol1, mol2, mask1 = None, mask2 = None):
    """
    Computes the similarity between two one-hot encoded molecules. The one-hot encoding indicates the atom type
    
    Args:
        mol1 (torch.Tensor): The first one-hot encoded molecule.
        mol2 (torch.Tensor): The second one-hot encoded molecule.
        mask (torch.Tensor, optional): A mask to apply on the atoms. Defaults to None.
    
    Returns:
        torch.Tensor: The similarity between the two molecules.
    """
    
    # Apply mask if provided
    if mask1 is not None:
        mask1 = mask1.bool()
        mol1 = mol1[mask1,:]

    if mask2 is not None:
        mask2 = mask2.bool()
        mol2 = mol2[mask2,:]
    
    # Compute similarity by comparing the one-hot encoded features
    similarity = torch.sum(mol1[:,3:-1] == mol2[:,3:-1]) / mol1[:, 3:-1].numel()
    
    return similarity

def compute_hausdorff_distance_batch(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the similarity between two molecules based on distances and atom type.
    
    Args:
        mol1 (torch.Tensor): The first molecule.
        mol2 (torch.Tensor): The second molecule.
        mask1 (torch.Tensor, optional): A mask indicating which atoms to consider for mo1. If not provided, all atoms will be considered.
        mask2 (torch.Tensor, optional): A mask indicating which atoms to consider for mol2. If not provided, all atoms will be considered.
        
    Returns:
        torch.Tensor: The similarity between the two molecules for each element in the batch.
    """
    # If fragment_mask is provided, only consider the atoms in the mask

    #take only the positions
    mol1 = mol1[:, :, :3]
    mol2 = mol2[:, :, :3]
    
    
    if mask1 is not None:
        mask1 = mask1.bool()
        batch_size = mol1.shape[0]
        masked_mol1 = []
        for i in range(batch_size):
            masked_mol1.append(mol1[i, mask1[i], :])
        
        if batch_size == 1:
            mol1 = masked_mol1[0].unsqueeze(0)
        else:    
            mol1 = torch.stack(masked_mol1)
        

    if mask2 is not None:
        mask2 = mask2.bool()
        mask2 = mask2.bool()
        batch_size = mol2.shape[0]
        masked_mol2 = []
        for i in range(batch_size):
            masked_mol2.append(mol2[i, mask2[i], :])
        
        if batch_size == 1:
            mol2 = masked_mol2[0].unsqueeze(0)
        else:    
            mol2 = torch.stack(masked_mol2)

    hausdorff_distances = []
    for i in range(mol1.shape[0]):
        hausdorff_distances.append(max(directed_hausdorff(mol1[i], mol2[i])[0], directed_hausdorff(mol2[i], mol1[i])[0]))

    return hausdorff_distances


def create_edge_index(mol, weighted=False):
    """
    Create edge index for a molecule.
    """
    adj = nx.to_scipy_sparse_array(mol).todense()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    if weighted:
        weights = torch.from_numpy(adj.data.astype(np.float32))
        edge_weight = torch.FloatTensor(weights)
        return edge_index, edge_weight

    return edge_index


def compute_coulomb_matrix(mol, mask=None, diagonalize=False):
    """
    Compute the Coulomb matrix for a molecule.
    
    Args:
        mol (torch.Tensor): The molecule tensor with shape (N, 4), where N is the number of atoms.
                            The last dimension should contain [x, y, z, atomic_number].
        mask (torch.Tensor, optional): A mask indicating which atoms to consider. If not provided, all atoms will be considered.
        diagonalize (bool, optional): Whether to return the diagonalized Coulomb matrix. Defaults to False.
        
    Returns:
        torch.Tensor: The Coulomb matrix of the molecule.
    """
    if mask is not None:
        mask = mask.bool()
        mol = mol[mask, :]

    positions = mol[:, :3]
    one_hot = mol[:, 3:]
    atomic_numbers = []
    
    for i, vec in enumerate(one_hot):
        if torch.sum(vec) == 1:
            atom_index = torch.argmax(vec).item()
            atomic_number = const.CHARGES[const.IDX2ATOM[atom_index]]
            atomic_numbers.append(atomic_number)
        else:
            atomic_numbers.append(0)  
    
    num_atoms = positions.shape[0]
    coulomb_matrix = torch.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                distance = torch.norm(positions[i] - positions[j])
                if distance == 0: #avoid division by zero
                    coulomb_matrix[i, j] = 0.0
                else:
                    coulomb_matrix[i, j] = atomic_numbers[i] * atomic_numbers[j] / distance

    if diagonalize:
        eigenvalues, eigenvectors = torch.linalg.eigh(coulomb_matrix)
        coulomb_matrix = torch.diag(eigenvalues)

    return coulomb_matrix

def compute_coulomb_matrices_batch(molecules, masks=None, diagonalize=False):
    """
    Compute the Coulomb matrices for a batch of molecules.
    
    Args:
        molecules (torch.Tensor): The batch of molecule tensors with shape (B, N, 4), where B is the batch size,
                                    N is the number of atoms, and the last dimension should contain [x, y, z, atomic_number].
        masks (torch.Tensor, optional): A batch of masks indicating which atoms to consider for each molecule. 
                                        If not provided, all atoms will be considered.
        
    Returns:
        torch.Tensor: The Coulomb matrices for the batch of molecules with shape (B, N, N).
    """
    batch_size = molecules.shape[0]
    # num_atoms = molecules.shape[1] #this is ok when the mask is not provided
    num_atoms = int(torch.sum(masks, dim=1).max().item()) if masks is not None else molecules.shape[1]
    coulomb_matrices = torch.zeros((batch_size, num_atoms, num_atoms), device=molecules.device)

    for b in range(batch_size):
        mol = molecules[b]
        mask = masks[b] if masks is not None else None
        coulomb_matrices[b] = compute_coulomb_matrix(mol, mask, diagonalize=diagonalize)

    return coulomb_matrices

def compute_frobenius_norm_batch(matrices):
    """
    Compute the Frobenius norm for a batch of matrices.
    
    Args:
        matrices (torch.Tensor): A batch of matrices with shape (B, N, N), where B is the batch size,
                                    and N is the number of rows/columns in each matrix.
        
    Returns:
        torch.Tensor: A tensor containing the Frobenius norm for each matrix in the batch.
    """
    # return torch.norm(matrices, dim=(1, 2), p='fro') #deprecated
    return torch.linalg.norm(matrices, ord='fro', dim=(1, 2))

def arrestomomentum():
    raise KeyboardInterrupt("Debug interrupt.")


# ## Explainability

# ### Utility function for visualization purposes

# In[5]:


def draw_sphere_xai(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) #* 0.8
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, alpha=alpha)

def plot_molecule_xai(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, is_geom, fragment_mask=None, phi_values=None, invert_colormap = False):
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
        phi_values_array = np.array(list(phi_values.values()))

        #draw fragments
        fragment_mask_on_cpu = fragment_mask.cpu().numpy()
        colors_fragment = colors[fragment_mask_on_cpu == 1]
        x_fragment = x[fragment_mask_on_cpu == 1]
        y_fragment = y[fragment_mask_on_cpu == 1]
        z_fragment = z[fragment_mask_on_cpu == 1]
        areas_fragment = areas[fragment_mask_on_cpu == 1]
        
        # Calculate the gradient colors based on phi values
        # cmap = plt.cm.get_cmap('coolwarm_r') #reversed heatmap for distance-based importance
        cmap = plt.cm.get_cmap('coolwarm') #heatmap for distance-based importance trying non reversed -> high shapley value mean more imporant, that drive the generation.
        #@mastro added invert_colormap to invert the colormap if average/expected value in higher than original prediction
        if invert_colormap:
            cmap = plt.cm.get_cmap('coolwarm_r')

        norm = plt.Normalize(vmin=min(phi_values_array), vmax=max(phi_values_array))
        colors_fragment_shadow = cmap(norm(phi_values_array))
        
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
                bg='black', alpha=1., fragment_mask=None, phi_values=None, invert_colormap = False):
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
        ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, is_geom=is_geom, fragment_mask=fragment_mask, phi_values=phi_values, invert_colormap=invert_colormap
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
        path, spheres_3d=False, bg="black", alpha=1.0, wandb=None, mode="chain", is_geom=False, fragment_mask=None, phi_values=None, invert_colormap = False
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
            invert_colormap=invert_colormap
        )
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


# ### Explainabiliy phase

# ##### Multiple sampling steps at a time

# In[ ]:


#@mastro
torch.set_printoptions(threshold=float('inf'))

num_samples = NUM_SAMPLES
sampled = 0
#end @mastro
start = 0

chain_with_full_fragments = None


# Create the folder if it does not exist
folder_save_path = f"results/explanations_coulomb_matrix_full_molecule_original_fragments_{P}_seed_{SEED}_{transformations_str}_transformation_seed_{TRANSFORMATION_SEED}"

if DIAGONALIZE:
    folder_save_path = "results/explanations_diagonalized_coulomb_" + P + "_seed_" + str(SEED) + "_full_molecule_original_fragments"
    
if not os.path.exists(folder_save_path):
    os.makedirs(folder_save_path)

data_list = []
for data in dataloader:

    if sampled < num_samples:
        data_list.append(data)
        sampled += 1

#determine max numebr of atoms of the molecules in the dataset. This is used to determine the size of the random noise, which we want to be equal for all molecules -> atoms not present in the molecule will be discarded using masks 
max_num_atoms = max(data["positions"].shape[1] for data in data_list)


#define initial random noise for positions and features #shape = [1, max_num_atoms, 3] for positions and [1, max_num_atoms, 8] for features. 1 since batch size is 1 for our explaination task
pos_size = (data_list[0]["positions"].shape[0], max_num_atoms, data_list[0]["positions"].shape[2])
feature_size = (data_list[0]["one_hot"].shape[0], max_num_atoms, data_list[0]["one_hot"].shape[2])

INTIAL_DISTIBUTION_PATH = "results/explanations_" + P + "_seed_" + str(SEED)
noisy_features = None
noisy_positions = None
#check if the initial distribution of the noisy features and positions already exists, if not create it
if os.path.exists(INTIAL_DISTIBUTION_PATH + "/noisy_features_seed_" + str(SEED) + ".pt"):
    # load initial distrubution of noisy features and positions
    print("Loading initial distribution of noisy features and positions.")
    noisy_features = torch.load(INTIAL_DISTIBUTION_PATH + "/noisy_features_seed_" + str(SEED) + ".pt", map_location=device, weights_only=True)
    noisy_positions = torch.load(INTIAL_DISTIBUTION_PATH + "/noisy_positions_seed_" + str(SEED) + ".pt", map_location=device, weights_only=True)

else:
    print("Creating initial distribution of noisy features and positions.")
    noisy_positions = torch.randn(pos_size, device=device)
    noisy_features = torch.randn(feature_size, device=device)


    #save the noisy positions and features on file .txt
    print("Saving noisy features and positions to .txt and .pt files.")
    noisy_positions_file = os.path.join(folder_save_path, "noisy_positions_seed_" + str(SEED) + ".txt")
    noisy_features_file = os.path.join(folder_save_path, "noisy_features_seed_" + str(SEED) + ".txt")

    with open(noisy_positions_file, "w") as f:
        f.write(str(noisy_positions))

    with open(noisy_features_file, "w") as f:
        f.write(str(noisy_features))

    torch.save(noisy_positions, os.path.join(folder_save_path, "noisy_positions_seed_" + str(SEED) + ".pt"))
    torch.save(noisy_features, os.path.join(folder_save_path, "noisy_features_seed_" + str(SEED) + ".pt"))

for data_index, data in enumerate(tqdm(data_list)): #7:

        # start_time = time.time()
        
        smile = data["name"][0]
        
        mol = read_smiles(smile)
        num_nodes = mol.number_of_nodes()
        
        num_edges = mol.number_of_edges()
        num_edges_directed = num_edges*2
        
        
        graph_density = num_edges_directed/(num_nodes*(num_nodes-1))
        max_number_of_nodes = num_edges + 1

        node_density = num_nodes/max_number_of_nodes

        node_edge_ratio = num_nodes/num_edges
        
        edge_node_ratio = num_edges/num_nodes
        
        if P == "graph_density":
            P = graph_density #probability of atom to exist in random graph
        elif P == "node_density":
            P = node_density
        elif P == "node_edge_ratio" or P == "edge_node_ratio":
            if node_edge_ratio < edge_node_ratio:
                P = node_edge_ratio
                print("Using node-edge ratio", node_edge_ratio)
            else:
                P = edge_node_ratio
                print("Using edge-node ratio", edge_node_ratio)            
        else:
            try:
                P = float(P)
            except ValueError:
                raise ValueError("P must be either 'graph_density', 'node_density', 'node_edge_ratio', 'edge_node_ratio' or a float value.")
        

        print("Using P:", P)

        chain_with_full_fragments = None
       
        rng = default_rng(seed = SEED)
        rng_torch = torch.Generator(device="cpu")
        rng_torch.manual_seed(SEED)

        #apply E(3) trasformations to the molecule. Linker atoms will be tranformed, too, but their transformations will be discarded in liue of the noisy positions
        print("Positions before transformations:", data["positions"])
        transform_rng = None
        if transformations:
            transform_rng = default_rng(seed = TRANSFORMATION_SEED)
            
        if ROTATE:
            #rotate molecule
            # Generate a random 3x3 matrix
            random_matrix = torch.tensor(transform_rng.uniform(-1, 1, (3, 3)), device=device, dtype=torch.float32)
            
            # Perform QR decomposition to obtain an orthogonal matrix
            q, r = torch.linalg.qr(random_matrix)
            
            # Ensure the determinant is 1 (if not, adjust it)
            if torch.det(q) < 0:
                q[:, 0] = -q[:, 0]
            
            #ensure q has float values
            # q = q.float()
            # Apply the rotation matrix to the molecule positions
            data["positions"] = torch.matmul(data["positions"], q)
        if TRANSLATE:
            #translate molecule
            translation_vector = torch.tensor(transform_rng.uniform(-1, 1, (1, 3)), device=device, dtype=torch.float32)
            data["positions"] = data["positions"] + translation_vector
        if REFLECT:
            #reflect molecule acrpss the xy plane
            reflection_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, -1.0]], device=device)
            data["positions"] = torch.matmul(data["positions"], reflection_matrix)
            

        print("Positions after transformations:", data["positions"])
        
        #filter the noisy positions and features to have the same size as the data, removing the atoms not actually present in the molecule
        #we use the same max sized noise for all molecules to guaranteethat the same moleclues are inzialized with the same noise for the linker atoms in common -> noise for the fragme atoms will be discarded
        noisy_positions_present_atoms = noisy_positions.clone()
        noisy_features_present_atoms = noisy_features.clone()

        noisy_positions_present_atoms = noisy_positions_present_atoms[:, :data["positions"].shape[1], :]
        noisy_features_present_atoms = noisy_features_present_atoms[:, :data["one_hot"].shape[1], :]

        # generate chain with original and full fragments
        chain_batch, node_mask = model.sample_chain(data, keep_frames=keep_frames, noisy_positions=noisy_positions_present_atoms, noisy_features=noisy_features_present_atoms)
        
        #get the generated molecule and store it in a variable
        chain_with_full_fragments = chain_batch[0, :, :, :] #need to get only the final frame, is 0 ok in the first dimension?
        
        #compute the Coulob matrix of the generated linker @mastro edited to try with full molecule to capute all the interactions
        # coulomb_matrix = compute_coulomb_matrix(chain_with_full_fragments.squeeze(), mask = data["linker_mask"][0].squeeze())
        
        #compute coulomb matrix for the whole molecule
        coulomb_matrix = compute_coulomb_matrix(chain_with_full_fragments.squeeze(), diagonalize=DIAGONALIZE)


        print("Coulomb matrix: ", coulomb_matrix)

        frobenius_norm_original_linker = torch.linalg.norm(coulomb_matrix, ord='fro')

        print("Frobenius norm of the Coulomb matrix original molecule:", frobenius_norm_original_linker)
        
    
        original_linker_mask_batch = data["linker_mask"][0].squeeze().repeat(PARALLEL_STEPS, 1) #check why it works
    
        
        
        num_fragment_atoms = torch.sum(data["fragment_mask"] == 1)

        phi_atoms = {}
        
        num_atoms = data["positions"].shape[1]
        num_linker_atoms = torch.sum(data["linker_mask"] == 1)
        
        # distances_random_samples = []
        # cosine_similarities_random_samples = []
        hausdorff_distances_random_samples = []
        frobenius_norm_random_samples = []
        # end_time = time.time()
        


        for j in tqdm(range(num_fragment_atoms)): 
            
            # marginal_contrib_distance = 0
            # marginal_contrib_cosine_similarity = 0
            # marginal_contrib_hausdorff = 0
            marginal_contrib_frobenius_norm = 0

            for step in range(int(M/PARALLEL_STEPS)):

                # start_time = time.time()

                fragment_indices = torch.where(data["fragment_mask"] == 1)[1]
                num_fragment_atoms = len(fragment_indices)
                fragment_indices = fragment_indices.repeat(PARALLEL_STEPS).to(device)

                data_j_plus = data.copy()
                data_j_minus = data.copy()
                data_random = data.copy()

                N_z_mask = torch.tensor(np.array([rng.binomial(1, P, size = num_fragment_atoms) for _ in range(PARALLEL_STEPS)]), dtype=torch.int32)
                # Ensure at least one element is 1, otherwise randomly select one since at least one fragment atom must be present
                
                
                for i in range(len(N_z_mask)):

                    #set the current explained atom to 0 in N_z_mask
                    N_z_mask[i][j] = 0 #so it is always one when taken from the oriignal sample and 0 when taken from the random sample. Check if it is more efficient to directly set it or check if it is already 0

                    if not N_z_mask[i].any():
                        
                        
                        random_index = j #j is the current explained atom, it should always be set to 0
                        while random_index == j:
                            random_index = rng.integers(0, num_fragment_atoms)
                        N_z_mask[i][random_index] = 1
                        
                       
                    

                N_z_mask=N_z_mask.flatten().to(device)
                
                
                

                N_mask = torch.ones(PARALLEL_STEPS * num_fragment_atoms, dtype=torch.int32, device=device)

                

                pi = torch.cat([torch.randperm(num_fragment_atoms, generator=rng_torch) for _ in range(PARALLEL_STEPS)], dim=0)

                N_j_plus_index = torch.ones(PARALLEL_STEPS*num_fragment_atoms, dtype=torch.int, device=device)
                N_j_minus_index = torch.ones(PARALLEL_STEPS*num_fragment_atoms, dtype=torch.int, device=device)

                selected_node_index = np.where(pi == j)
                selected_node_index = torch.tensor(np.array(selected_node_index), device=device).squeeze()
                selected_node_index = selected_node_index.repeat_interleave(num_fragment_atoms) #@mastro TO BE CHECKED IF THIS IS CORRECT
                
                k_values = torch.arange(num_fragment_atoms*PARALLEL_STEPS, device=device)

                add_to_pi = torch.arange(start=0, end=PARALLEL_STEPS*num_fragment_atoms, step=num_fragment_atoms).repeat_interleave(num_fragment_atoms) #check if it is correct ot consider num_fragment_atoms and not num_atoms

                pi_add = pi + add_to_pi
                pi_add = pi_add.to(device=device)
                #this must be cafeully checked. this should be adapted for nodes
                add_to_node_index = torch.arange(start=0, end=PARALLEL_STEPS*num_atoms, step=num_atoms) #@mastro change step from num_fragment_atoms to num_atoms
                
                add_to_node_index = add_to_node_index.repeat_interleave(num_fragment_atoms).to(device) #changed from num_atoms to num_fragment_atoms

                
                N_j_plus_index[pi_add] = torch.where(k_values <= selected_node_index, N_mask[pi_add], N_z_mask[pi_add])
                N_j_minus_index[pi_add] = torch.where(k_values < selected_node_index, N_mask[pi_add], N_z_mask[pi_add]) 

                #fragements to keep in molecule j plus
                fragment_indices = fragment_indices + add_to_node_index
                
                
                N_j_plus = fragment_indices[(N_j_plus_index==1)] #fragment to keep in molecule j plus
                #fragement indices to keep in molecule j minus
               
                N_j_minus = fragment_indices[(N_j_minus_index==1)] #it is ok. it contains fragmens indices to keep in molecule j minus (indices that index the atom nodes)

                #fragement indices to keep in random molecule
                N_random_sample = fragment_indices[(N_z_mask==1)] 
                
               
                atom_mask_j_plus = torch.zeros(num_atoms*PARALLEL_STEPS, dtype=torch.bool)
                atom_mask_j_minus = torch.zeros(num_atoms*PARALLEL_STEPS, dtype=torch.bool)
                atom_mask_random_molecule = torch.zeros(num_atoms*PARALLEL_STEPS, dtype=torch.bool)

                atom_mask_j_plus[N_j_plus] = True
                
                atom_mask_j_minus[N_j_minus] = True

                #set to true also linker atoms
                parallelized_linker_mask = data["linker_mask"][0].squeeze().to(torch.int).repeat(PARALLEL_STEPS)
                atom_mask_j_plus[(parallelized_linker_mask == 1)] = True 

                #set to true also linker atoms
                atom_mask_j_minus[(parallelized_linker_mask == 1)] = True 

                atom_mask_random_molecule[N_random_sample] = True
                #set to true also linker atoms
                atom_mask_random_molecule[(parallelized_linker_mask == 1)] = True
                
               
                atom_mask_j_plus = atom_mask_j_plus.view(PARALLEL_STEPS, num_atoms)
                atom_mask_j_minus = atom_mask_j_minus.view(PARALLEL_STEPS, num_atoms)
                atom_mask_random_molecule = atom_mask_random_molecule.view(PARALLEL_STEPS, num_atoms)
                
                

                data_j_plus_dict = {}
                data_j_minus_dict = {}
                data_random_dict = {}

                noisy_features_j_plus_dict = {}
                noisy_positions_j_plus_dict = {}
                noisy_features_j_minus_dict = {}
                noisy_positions_j_minus_dict = {}
                noisy_features_random_dict = {}
                noisy_positions_random_dict = {}
                
                # start_time = time.time()
                for i in range(PARALLEL_STEPS):

                    # Remove fragment atoms that are not present for j plus
                    noisy_features_present_atoms_j_plus = noisy_features_present_atoms.clone()
                    noisy_features_j_plus_dict[i] = noisy_features_present_atoms_j_plus[:, atom_mask_j_plus[i], :]
                    
                    noisy_positions_present_atoms_j_plus = noisy_positions_present_atoms.clone()
                    noisy_positions_j_plus_dict[i] = noisy_positions_present_atoms_j_plus[:, atom_mask_j_plus[i], :]

                    # Remove fragment atoms that are not present for j minus
                    noisy_features_present_atoms_j_minus = noisy_features_present_atoms.clone()
                    noisy_features_j_minus_dict[i] = noisy_features_present_atoms_j_minus[:, atom_mask_j_minus[i], :]

                    noisy_positions_present_atoms_j_minus = noisy_positions_present_atoms.clone()
                    noisy_positions_j_minus_dict[i] = noisy_positions_present_atoms_j_minus[:, atom_mask_j_minus[i], :]

                    # Remove fragment atoms that are not present for random molecule
                    noisy_features_present_atoms_random = noisy_features_present_atoms.clone()
                    noisy_features_random_dict[i] = noisy_features_present_atoms_random[:, atom_mask_random_molecule[i], :]

                    noisy_positions_present_atoms_random = noisy_positions_present_atoms.clone()
                    noisy_positions_random_dict[i] = noisy_positions_present_atoms_random[:, atom_mask_random_molecule[i], :]


                    data_j_plus_dict[i] = data.copy()
                    data_j_minus_dict[i] = data.copy()
                    data_random_dict[i] = data.copy()

                    #data j plus
                    data_j_plus_dict[i]["positions"] = data_j_plus_dict[i]["positions"][:, atom_mask_j_plus[i]]
                    data_j_plus_dict[i]["num_atoms"] = data_j_plus_dict[i]["positions"].shape[1]
                    # remove one_hot of atoms in random_indices
                    data_j_plus_dict[i]["one_hot"] = data_j_plus_dict[i]["one_hot"][:, atom_mask_j_plus[i]]
                    # remove atom_mask of atoms in random_indices
                    data_j_plus_dict[i]["atom_mask"] = data_j_plus_dict[i]["atom_mask"][:, atom_mask_j_plus[i]]
                    # remove fragment_mask of atoms in random_indices
                    data_j_plus_dict[i]["fragment_mask"] = data_j_plus_dict[i]["fragment_mask"][:, atom_mask_j_plus[i]]
                    # remove linker_mask of atoms in random_indices
                    data_j_plus_dict[i]["linker_mask"] = data_j_plus_dict[i]["linker_mask"][:, atom_mask_j_plus[i]]
                    data_j_plus_dict[i]["charges"] = data_j_plus_dict[i]["charges"][:, atom_mask_j_plus[i]]
                    data_j_plus_dict[i]["anchors"] = data_j_plus_dict[i]["anchors"][:, atom_mask_j_plus[i]]
                    edge_mask_to_keep = (atom_mask_j_plus[i].unsqueeze(1) * atom_mask_j_plus[i]).flatten()
                    data_j_plus_dict[i]["edge_mask"] = data_j_plus_dict[i]["edge_mask"][edge_mask_to_keep]

                    #data j minus
                    data_j_minus_dict[i]["positions"] = data_j_minus_dict[i]["positions"][:, atom_mask_j_minus[i]]
                    data_j_minus_dict[i]["num_atoms"] = data_j_minus_dict[i]["positions"].shape[1]
                    # remove one_hot of atoms in random_indices
                    data_j_minus_dict[i]["one_hot"] = data_j_minus_dict[i]["one_hot"][:, atom_mask_j_minus[i]]
                    # remove atom_mask of atoms in random_indices
                    data_j_minus_dict[i]["atom_mask"] = data_j_minus_dict[i]["atom_mask"][:, atom_mask_j_minus[i]]
                    # remove fragment_mask of atoms in random_indices
                    data_j_minus_dict[i]["fragment_mask"] = data_j_minus_dict[i]["fragment_mask"][:, atom_mask_j_minus[i]]
                    # remove linker_mask of atoms in random_indices
                    data_j_minus_dict[i]["linker_mask"] = data_j_minus_dict[i]["linker_mask"][:, atom_mask_j_minus[i]]
                    data_j_minus_dict[i]["charges"] = data_j_minus_dict[i]["charges"][:, atom_mask_j_minus[i]]
                    data_j_minus_dict[i]["anchors"] = data_j_minus_dict[i]["anchors"][:, atom_mask_j_minus[i]]
                    # remove edge_mask of atoms in random_indices
                    edge_mask_to_keep = (atom_mask_j_minus[i].unsqueeze(1) * atom_mask_j_minus[i]).flatten() 
                    data_j_minus_dict[i]["edge_mask"] = data_j_minus_dict[i]["edge_mask"][edge_mask_to_keep]

                    #data random
                    data_random_dict[i]["positions"] = data_random_dict[i]["positions"][:, atom_mask_random_molecule[i]]
                    data_random_dict[i]["num_atoms"] = data_random_dict[i]["positions"].shape[1]
                    # remove one_hot of atoms in random_indices
                    data_random_dict[i]["one_hot"] = data_random_dict[i]["one_hot"][:, atom_mask_random_molecule[i]]
                    # remove atom_mask of atoms in random_indices
                    data_random_dict[i]["atom_mask"] = data_random_dict[i]["atom_mask"][:, atom_mask_random_molecule[i]]
                    # remove fragment_mask of atoms in random_indices
                    data_random_dict[i]["fragment_mask"] = data_random_dict[i]["fragment_mask"][:, atom_mask_random_molecule[i]]
                    # remove linker_mask of atoms in random_indices
                    data_random_dict[i]["linker_mask"] = data_random_dict[i]["linker_mask"][:, atom_mask_random_molecule[i]]
                    data_random_dict[i]["charges"] = data_random_dict[i]["charges"][:, atom_mask_random_molecule[i]]
                    data_random_dict[i]["anchors"] = data_random_dict[i]["anchors"][:, atom_mask_random_molecule[i]]
                    # remove edge_mask of atoms in random_indices
                    # remove edge_mask of atoms in random_indices
                    edge_mask_to_keep = (atom_mask_random_molecule[i].unsqueeze(1) * atom_mask_random_molecule[i]).flatten() 

                    data_random_dict[i]["edge_mask"] = data_random_dict[i]["edge_mask"][edge_mask_to_keep]
                
                

                PADDING = True

                # start_time = time.time()
                if PADDING:

                    max_atoms_j_plus = max(data_j_plus_dict[i]["num_atoms"] for i in range(PARALLEL_STEPS))

                    max_edges_j_plus = max(data_j_plus_dict[i]["edge_mask"].shape[0] for i in range(PARALLEL_STEPS))
                    
                    
                    max_atoms_j_minus = max(data_j_minus_dict[i]["num_atoms"] for i in range(PARALLEL_STEPS))

                    max_edges_j_minus = max(data_j_minus_dict[i]["edge_mask"].shape[0] for i in range(PARALLEL_STEPS))

                    max_atoms_random = max(data_random_dict[i]["num_atoms"] for i in range(PARALLEL_STEPS))

                    max_edges_random = max(data_random_dict[i]["edge_mask"].shape[0] for i in range(PARALLEL_STEPS))
                    
                    for i in range(PARALLEL_STEPS):
                        #for j plus positions
                        num_atoms_to_stack = max_atoms_j_plus - data_j_plus_dict[i]["positions"].shape[1]
                        padding = torch.zeros(data_j_plus_dict[i]["positions"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["positions"].shape[2]).to(device)
                        stacked_positions = torch.cat((data_j_plus_dict[i]["positions"], padding), dim=1)
                        data_j_plus_dict[i]["positions"] = stacked_positions
                        #for j plus one_hot
                        padding = torch.zeros(data_j_plus_dict[i]["one_hot"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["one_hot"].shape[2]).to(device)
                        stacked_one_hot = torch.cat((data_j_plus_dict[i]["one_hot"], padding), dim=1)
                        data_j_plus_dict[i]["one_hot"] = stacked_one_hot
                        padding = torch.zeros(data_j_plus_dict[i]["fragment_mask"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["fragment_mask"].shape[2]).to(device)
                        stacked_fragment_mask = torch.cat((data_j_plus_dict[i]["fragment_mask"], padding), dim=1)
                        data_j_plus_dict[i]["fragment_mask"] = stacked_fragment_mask
                        padding = torch.zeros(data_j_plus_dict[i]["charges"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["charges"].shape[2]).to(device)
                        stacked_charges = torch.cat((data_j_plus_dict[i]["charges"], padding), dim=1)
                        data_j_plus_dict[i]["charges"] = stacked_charges
                        padding = torch.zeros(data_j_plus_dict[i]["anchors"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["anchors"].shape[2]).to(device)
                        stacked_anchors = torch.cat((data_j_plus_dict[i]["anchors"], padding), dim=1)
                        data_j_plus_dict[i]["anchors"] = stacked_anchors
                        padding = torch.zeros(data_j_plus_dict[i]["linker_mask"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["linker_mask"].shape[2]).to(device)
                        stacked_linker_mask = torch.cat((data_j_plus_dict[i]["linker_mask"], padding), dim=1)
                        data_j_plus_dict[i]["linker_mask"] = stacked_linker_mask
                        padding = torch.zeros(data_j_plus_dict[i]["atom_mask"].shape[0], num_atoms_to_stack, data_j_plus_dict[i]["atom_mask"].shape[2]).to(device)
                        stacked_atom_mask = torch.cat((data_j_plus_dict[i]["atom_mask"], padding), dim=1)
                        data_j_plus_dict[i]["atom_mask"] = stacked_atom_mask
                        num_edges_to_stack = max_edges_j_plus - data_j_plus_dict[i]["edge_mask"].shape[0]
                        data_j_plus_dict[i]["edge_mask"] = data_j_plus_dict[i]["edge_mask"].unsqueeze(0)
                        padding = torch.zeros(data_j_plus_dict[i]["edge_mask"].shape[0], num_edges_to_stack, data_j_plus_dict[i]["edge_mask"].shape[2]).to(device)
                        stacked_edge_mask = torch.cat((data_j_plus_dict[i]["edge_mask"], padding), dim=1)
                        data_j_plus_dict[i]["edge_mask"] = stacked_edge_mask
                        
                        #for noisy positions and features for j plus
                        noisy_positions_j_plus_dict[i] = noisy_positions_j_plus_dict[i] #check this
                        padding = torch.zeros(noisy_positions_j_plus_dict[i].shape[0], num_atoms_to_stack, noisy_positions_j_plus_dict[i].shape[2]).to(device)
                        stacked_positions = torch.cat((noisy_positions_j_plus_dict[i], padding), dim=1)
                        noisy_positions_j_plus_dict[i] = stacked_positions

                        noisy_features_j_plus_dict[i] = noisy_features_j_plus_dict[i]
                        padding = torch.zeros(noisy_features_j_plus_dict[i].shape[0], num_atoms_to_stack, noisy_features_j_plus_dict[i].shape[2]).to(device)
                        stacked_features = torch.cat((noisy_features_j_plus_dict[i], padding), dim=1)
                        noisy_features_j_plus_dict[i] = stacked_features

                        #for j minus
                        num_atoms_to_stack = max_atoms_j_minus - data_j_minus_dict[i]["positions"].shape[1]
                        padding = torch.zeros(data_j_minus_dict[i]["positions"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["positions"].shape[2]).to(device) #why does this work?
                        stacked_positions = torch.cat((data_j_minus_dict[i]["positions"], padding), dim=1)
                        data_j_minus_dict[i]["positions"] = stacked_positions
                        
                        padding = torch.zeros(data_j_minus_dict[i]["one_hot"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["one_hot"].shape[2]).to(device)
                        stacked_one_hot = torch.cat((data_j_minus_dict[i]["one_hot"], padding), dim=1)
                        data_j_minus_dict[i]["one_hot"] = stacked_one_hot
                        
                        padding = torch.zeros(data_j_minus_dict[i]["fragment_mask"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["fragment_mask"].shape[2]).to(device)
                        stacked_fragment_mask = torch.cat((data_j_minus_dict[i]["fragment_mask"], padding), dim=1)
                        data_j_minus_dict[i]["fragment_mask"] = stacked_fragment_mask

                        
                        padding = torch.zeros(data_j_minus_dict[i]["charges"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["charges"].shape[2]).to(device)
                        stacked_charges = torch.cat((data_j_minus_dict[i]["charges"], padding), dim=1)
                        data_j_minus_dict[i]["charges"] = stacked_charges
                        
                        padding = torch.zeros(data_j_minus_dict[i]["anchors"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["anchors"].shape[2]).to(device)
                        stacked_anchors = torch.cat((data_j_minus_dict[i]["anchors"], padding), dim=1)
                        data_j_minus_dict[i]["anchors"] = stacked_anchors
                       
                        padding = torch.zeros(data_j_minus_dict[i]["linker_mask"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["linker_mask"].shape[2]).to(device)
                        stacked_linker_mask = torch.cat((data_j_minus_dict[i]["linker_mask"], padding), dim=1)
                        data_j_minus_dict[i]["linker_mask"] = stacked_linker_mask
                        
                        padding = torch.zeros(data_j_minus_dict[i]["atom_mask"].shape[0], num_atoms_to_stack, data_j_minus_dict[i]["atom_mask"].shape[2]).to(device)
                        stacked_atom_mask = torch.cat((data_j_minus_dict[i]["atom_mask"], padding), dim=1)
                        data_j_minus_dict[i]["atom_mask"] = stacked_atom_mask
                        
                        num_edges_to_stack = max_edges_j_minus - data_j_minus_dict[i]["edge_mask"].shape[0]
                        data_j_minus_dict[i]["edge_mask"] = data_j_minus_dict[i]["edge_mask"].unsqueeze(0)
                        padding = torch.zeros(data_j_minus_dict[i]["edge_mask"].shape[0], num_edges_to_stack, data_j_minus_dict[i]["edge_mask"].shape[2]).to(device)
                        stacked_edge_mask = torch.cat((data_j_minus_dict[i]["edge_mask"], padding), dim=1)
                        data_j_minus_dict[i]["edge_mask"] = stacked_edge_mask
                    
                        #for noisy positions and features for j plus
                        noisy_positions_j_minus_dict[i] = noisy_positions_j_minus_dict[i] #check this
                        padding = torch.zeros(noisy_positions_j_minus_dict[i].shape[0], num_atoms_to_stack, noisy_positions_j_minus_dict[i].shape[2]).to(device)
                        stacked_positions = torch.cat((noisy_positions_j_minus_dict[i], padding), dim=1)
                        noisy_positions_j_minus_dict[i] = stacked_positions

                        noisy_features_j_minus_dict[i] = noisy_features_j_minus_dict[i]
                        padding = torch.zeros(noisy_features_j_minus_dict[i].shape[0], num_atoms_to_stack, noisy_features_j_minus_dict[i].shape[2]).to(device)
                        stacked_features = torch.cat((noisy_features_j_minus_dict[i], padding), dim=1)
                        noisy_features_j_minus_dict[i] = stacked_features

                        #for random
                        num_atoms_to_stack = max_atoms_random - data_random_dict[i]["positions"].shape[1]
                        padding = torch.zeros(data_random_dict[i]["positions"].shape[0], num_atoms_to_stack, data_random_dict[i]["positions"].shape[2]).to(device)
                        stacked_positions = torch.cat((data_random_dict[i]["positions"], padding), dim=1)
                        data_random_dict[i]["positions"] = stacked_positions
                        
                        padding = torch.zeros(data_random_dict[i]["one_hot"].shape[0], num_atoms_to_stack, data_random_dict[i]["one_hot"].shape[2]).to(device)
                        stacked_one_hot = torch.cat((data_random_dict[i]["one_hot"], padding), dim=1)
                        data_random_dict[i]["one_hot"] = stacked_one_hot
                        
                        padding = torch.zeros(data_random_dict[i]["fragment_mask"].shape[0], num_atoms_to_stack, data_random_dict[i]["fragment_mask"].shape[2]).to(device)
                        stacked_fragment_mask = torch.cat((data_random_dict[i]["fragment_mask"], padding), dim=1)
                        data_random_dict[i]["fragment_mask"] = stacked_fragment_mask
                        
                        padding = torch.zeros(data_random_dict[i]["linker_mask"].shape[0], num_atoms_to_stack, data_random_dict[i]["linker_mask"].shape[2]).to(device)
                        stacked_linker_mask = torch.cat((data_random_dict[i]["linker_mask"], padding), dim=1)
                        data_random_dict[i]["linker_mask"] = stacked_linker_mask

                       
                        padding = torch.zeros(data_random_dict[i]["charges"].shape[0], num_atoms_to_stack, data_random_dict[i]["charges"].shape[2]).to(device)
                        stacked_charges = torch.cat((data_random_dict[i]["charges"], padding), dim=1)
                        data_random_dict[i]["charges"] = stacked_charges

                    
                        padding = torch.zeros(data_random_dict[i]["anchors"].shape[0], num_atoms_to_stack, data_random_dict[i]["anchors"].shape[2]).to(device)
                        stacked_anchors = torch.cat((data_random_dict[i]["anchors"], padding), dim=1)
                        data_random_dict[i]["anchors"] = stacked_anchors
                       
                        padding = torch.zeros(data_random_dict[i]["atom_mask"].shape[0], num_atoms_to_stack, data_random_dict[i]["atom_mask"].shape[2]).to(device)
                        stacked_atom_mask = torch.cat((data_random_dict[i]["atom_mask"], padding), dim=1)
                        data_random_dict[i]["atom_mask"] = stacked_atom_mask
                        
                        num_edges_to_stack = max_edges_random - data_random_dict[i]["edge_mask"].shape[0]
                        data_random_dict[i]["edge_mask"] = data_random_dict[i]["edge_mask"].unsqueeze(0)
                        padding = torch.zeros(data_random_dict[i]["edge_mask"].shape[0], num_edges_to_stack, data_random_dict[i]["edge_mask"].shape[2]).to(device)
                        stacked_edge_mask = torch.cat((data_random_dict[i]["edge_mask"], padding), dim=1)
                        data_random_dict[i]["edge_mask"] = stacked_edge_mask

                        #for noisy positions and features for j plus
                        noisy_positions_random_dict[i] = noisy_positions_random_dict[i] #check this
                        padding = torch.zeros(noisy_positions_random_dict[i].shape[0], num_atoms_to_stack, noisy_positions_random_dict[i].shape[2]).to(device)
                        stacked_positions = torch.cat((noisy_positions_random_dict[i], padding), dim=1)
                        noisy_positions_random_dict[i] = stacked_positions

                        noisy_features_random_dict[i] = noisy_features_random_dict[i]
                        padding = torch.zeros(noisy_features_random_dict[i].shape[0], num_atoms_to_stack, noisy_features_random_dict[i].shape[2]).to(device)
                        stacked_features = torch.cat((noisy_features_random_dict[i], padding), dim=1)
                        noisy_features_random_dict[i] = stacked_features
                        
                        

                
                #create batch for j plus
                data_j_plus_batch = {}
                data_j_plus_batch["positions"] = torch.stack([data_j_plus_dict[i]["positions"] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                data_j_plus_batch["one_hot"] = torch.stack([data_j_plus_dict[i]["one_hot"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_plus_batch["atom_mask"] = torch.stack([data_j_plus_dict[i]["atom_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_plus_batch["fragment_mask"] = torch.stack([data_j_plus_dict[i]["fragment_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_plus_batch["linker_mask"] = torch.stack([data_j_plus_dict[i]["linker_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_plus_batch["charges"] = torch.stack([data_j_plus_dict[i]["charges"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_plus_batch["anchors"] = torch.stack([data_j_plus_dict[i]["anchors"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                
                
                data_j_plus_batch["uuid"] = [i for i in range(PARALLEL_STEPS)]
                data_j_plus_batch["num_atoms"] = [data_j_plus_dict[i]["num_atoms"] for i in range(PARALLEL_STEPS)]
                data_j_plus_batch["name"] = [data["name"] for _ in range(PARALLEL_STEPS)]
                data_j_plus_batch["edge_mask"] = torch.cat([data_j_plus_dict[i]["edge_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze().view(-1).unsqueeze(1)


                #create batch for j minus
                data_j_minus_batch = {}
                data_j_minus_batch["positions"] = torch.stack([data_j_minus_dict[i]["positions"] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                data_j_minus_batch["one_hot"] = torch.stack([data_j_minus_dict[i]["one_hot"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_minus_batch["atom_mask"] = torch.stack([data_j_minus_dict[i]["atom_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_minus_batch["fragment_mask"] = torch.stack([data_j_minus_dict[i]["fragment_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_minus_batch["linker_mask"] = torch.stack([data_j_minus_dict[i]["linker_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_minus_batch["charges"] = torch.stack([data_j_minus_dict[i]["charges"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_j_minus_batch["anchors"] = torch.stack([data_j_minus_dict[i]["anchors"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                
                data_j_minus_batch["uuid"] = [i for i in range(PARALLEL_STEPS)]
                data_j_minus_batch["num_atoms"] = [data_j_minus_dict[i]["num_atoms"] for i in range(PARALLEL_STEPS)]
                data_j_minus_batch["name"] = [data["name"] for _ in range(PARALLEL_STEPS)]
                data_j_minus_batch["edge_mask"] = torch.cat([data_j_minus_dict[i]["edge_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze().view(-1).unsqueeze(1)

                #create batch for random
                data_random_batch = {}
                data_random_batch["positions"] = torch.stack([data_random_dict[i]["positions"] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                data_random_batch["one_hot"] = torch.stack([data_random_dict[i]["one_hot"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_random_batch["atom_mask"] = torch.stack([data_random_dict[i]["atom_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_random_batch["fragment_mask"] = torch.stack([data_random_dict[i]["fragment_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_random_batch["linker_mask"] = torch.stack([data_random_dict[i]["linker_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_random_batch["charges"] = torch.stack([data_random_dict[i]["charges"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                data_random_batch["anchors"] = torch.stack([data_random_dict[i]["anchors"] for i in range(PARALLEL_STEPS)], dim=0).squeeze(1)
                
                data_random_batch["uuid"] = [i for i in range(PARALLEL_STEPS)]
                data_random_batch["num_atoms"] = [data_random_dict[i]["num_atoms"] for i in range(PARALLEL_STEPS)]
                data_random_batch["name"] = [data["name"] for _ in range(PARALLEL_STEPS)]
                data_random_batch["edge_mask"] = torch.cat([data_random_dict[i]["edge_mask"] for i in range(PARALLEL_STEPS)], dim=0).squeeze().view(-1).unsqueeze(1)

                
                #create batches for noisy positions and features
                noisy_positions_batch_j_plus = torch.stack([noisy_positions_j_plus_dict[i] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                noisy_features_batch_j_plus = torch.stack([noisy_features_j_plus_dict[i] for i in range(PARALLEL_STEPS)], dim=0).squeeze()

                noisy_positions_batch_j_minus = torch.stack([noisy_positions_j_minus_dict[i] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                noisy_features_batch_j_minus = torch.stack([noisy_features_j_minus_dict[i] for i in range(PARALLEL_STEPS)], dim=0).squeeze()

                noisy_positions_batch_random = torch.stack([noisy_positions_random_dict[i] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                noisy_features_batch_random = torch.stack([noisy_features_random_dict[i] for i in range(PARALLEL_STEPS)], dim=0).squeeze()
                
                

                
                chain_j_plus_batch, node_mask_j_plus_batch = model.sample_chain(data_j_plus_batch, keep_frames=keep_frames, noisy_positions=noisy_positions_batch_j_plus, noisy_features=noisy_features_batch_j_plus)

                chain_j_plus = chain_j_plus_batch[0, :, :, :] #it should take the first frame and all batch elements -> check it is really the first frame (I need the one at t0, the final generated molecule)
                

                chain_j_minus_batch, node_mask_j_minus_batch = model.sample_chain(data_j_minus_batch, keep_frames=keep_frames, noisy_positions=noisy_positions_batch_j_minus, noisy_features=noisy_features_batch_j_minus)

                chain_j_minus = chain_j_minus_batch[0, :, :, :]

                chain_random_batch, node_mask_random_batch = model.sample_chain(data_random_batch, keep_frames=keep_frames, noisy_positions=noisy_positions_batch_random, noisy_features=noisy_features_batch_random)

                chain_random = chain_random_batch[0, :, :, :]
                
                

                chain_with_full_fragments_batch = chain_with_full_fragments.repeat(PARALLEL_STEPS, 1, 1)

                # Check if all vectors in data_j_plus_batch["linker_mask"] are the same
                
                
                ####NEW CODE#######
                #@mastro creating new molecule containing the original fragmsnes and the linker generated using molecule j_plus, molecule j_minus and random molecule
                chain_j_plus_batch_original_fragments = chain_with_full_fragments_batch.clone()
                
                # Ensure the masks have the correct shape
                mask1 = data["linker_mask"][0].squeeze() == 1
                mask2 = data_j_plus_batch["linker_mask"].squeeze() == 1

                # Check if the masks need to be expanded
                if mask1.dim() == 1 and chain_j_plus_batch_original_fragments.dim() == 3:
                    mask1 = mask1.unsqueeze(0).expand(chain_j_plus_batch_original_fragments.size(0), -1)

                
                # Apply the masks
                # Ensure the shapes match for the assignment
                if chain_j_plus_batch_original_fragments[mask1, :].shape == chain_j_plus[mask2, :].shape:
                    chain_j_plus_batch_original_fragments[mask1, :] = chain_j_plus[mask2, :]
                else:
                    print("Shape mismatch:", chain_j_plus_batch_original_fragments[mask1, :].shape, chain_j_plus[mask2, :].shape)
                
                
                # print("chain_j_plus_batch_original_fragments shape", chain_j_plus_batch_original_fragments.shape)

                
                # chain_j_minus_batch_original_fragments = chain_j_minus.clone()
                # chain_j_minus_batch_original_fragments[:, data["fragment_mask"][0].squeeze() == 1, :] = chain_with_full_fragments_batch[:, data["fragment_mask"][0].squeeze() == 1, :]

                chain_j_minus_batch_original_fragments = chain_with_full_fragments_batch.clone()
                
                # Ensure the masks have the correct shape
                mask1 = data["linker_mask"][0].squeeze() == 1
                mask2 = data_j_minus_batch["linker_mask"].squeeze() == 1

                # Check if the masks need to be expanded
                if mask1.dim() == 1 and chain_j_minus_batch_original_fragments.dim() == 3:
                    mask1 = mask1.unsqueeze(0).expand(chain_j_minus_batch_original_fragments.size(0), -1)

                
                # Apply the masks
                # Ensure the shapes match for the assignment
                if chain_j_minus_batch_original_fragments[mask1, :].shape == chain_j_minus[mask2, :].shape:
                    chain_j_minus_batch_original_fragments[mask1, :] = chain_j_minus[mask2, :]
                else:
                    print("Shape mismatch:", chain_j_minus_batch_original_fragments[mask1, :].shape, chain_j_minus[mask2, :].shape)
                
                
                # print("chain_j_minus_batch_original_fragments shape", chain_j_minus_batch_original_fragments.shape)

                # chain_random_batch_original_fragments = chain_random.clone()
                # chain_random_batch_original_fragments[:, data["fragment_mask"][0].squeeze() == 1, :] = chain_with_full_fragments_batch[:, data["fragment_mask"][0].squeeze() == 1, :]

                # chain_random_batch_original_fragments = chain_with_full_fragments_batch.clone()
                # chain_random_batch_original_fragments[data["linker_mask"][0].squeeze() == 1, :] = chain_random[:, data_random_batch["linker_mask"].squeeze() == 1, :]

                chain_random_batch_original_fragments = chain_with_full_fragments_batch.clone()
                
                # Ensure the masks have the correct shape
                mask1 = data["linker_mask"][0].squeeze() == 1
                mask2 = data_random_batch["linker_mask"].squeeze() == 1

                # Check if the masks need to be expanded
                if mask1.dim() == 1 and chain_random_batch_original_fragments.dim() == 3:
                    mask1 = mask1.unsqueeze(0).expand(chain_random_batch_original_fragments.size(0), -1)

                
                # Apply the masks
                # Ensure the shapes match for the assignment
                if chain_random_batch_original_fragments[mask1, :].shape == chain_random[mask2, :].shape:
                    chain_random_batch_original_fragments[mask1, :] = chain_random[mask2, :]
                else:
                    print("Shape mismatch:", chain_random_batch_original_fragments[mask1, :].shape, chain_random[mask2, :].shape)
                
                
                # print("chain_random_batch_original_fragments shape", chain_random_batch_original_fragments.shape)
                ###################################


                # V_j_plus_coulomb_matrices_batch = compute_coulomb_matrices_batch(chain_j_plus.cpu())

                V_j_plus_coulomb_matrices_batch = compute_coulomb_matrices_batch(chain_j_plus_batch_original_fragments.cpu(), diagonalize=DIAGONALIZE)
                
                V_j_plus_frobenius_norm_batch = compute_frobenius_norm_batch(V_j_plus_coulomb_matrices_batch)
                
                V_j_plus_frobenius_norm = sum(V_j_plus_frobenius_norm_batch)
                
                print("V_j_plus_frobenius_norm", V_j_plus_frobenius_norm)

                #non diagonalized version for testing
                V_j_plus_coulomb_matrices_batch_non_diag = compute_coulomb_matrices_batch(chain_j_plus_batch_original_fragments.cpu(), diagonalize=False)

                V_j_plus_frobenius_norm_batch_non_diag = compute_frobenius_norm_batch(V_j_plus_coulomb_matrices_batch_non_diag)

                V_j_plus_frobenius_norm_non_diag = sum(V_j_plus_frobenius_norm_batch_non_diag)

                print("V_j_plus_frobenius_norm_non_diag", V_j_plus_frobenius_norm_non_diag)
                # print("V_j_plus_frobenius_norm", V_j_plus_frobenius_norm)
                #@mastro computing for the whole molecule
                # V_j_minus_coulomb_matrices_batch = compute_coulomb_matrices_batch(chain_j_minus.cpu())
                
                V_j_minus_coulomb_matrices_batch = compute_coulomb_matrices_batch(chain_j_minus_batch_original_fragments.cpu(), diagonalize=DIAGONALIZE)

                V_j_minus_frobenius_norm_batch = compute_frobenius_norm_batch(V_j_minus_coulomb_matrices_batch)

                V_j_minus_frobenius_norm = sum(V_j_minus_frobenius_norm_batch)

                print("V_j_minus_frobenius_norm", V_j_minus_frobenius_norm)

                #non diagonalized version for testing
                V_j_minus_coulomb_matrices_batch_non_diag = compute_coulomb_matrices_batch(chain_j_minus_batch_original_fragments.cpu(), diagonalize=False)

                V_j_minus_frobenius_norm_batch_non_diag = compute_frobenius_norm_batch(V_j_minus_coulomb_matrices_batch_non_diag)

                V_j_minus_frobenius_norm_non_diag = sum(V_j_minus_frobenius_norm_batch_non_diag)

                print("V_j_minus_frobenius_norm_non_diag", V_j_minus_frobenius_norm_non_diag)
                
                # print("V_j_minus_frobenius_norm", V_j_minus_frobenius_norm)

                # V_random_coulomb_matrices_batch = compute_coulomb_matrices_batch(chain_random.cpu())

                V_random_coulomb_matrices_batch = compute_coulomb_matrices_batch(chain_random_batch_original_fragments.cpu(), diagonalize=DIAGONALIZE)

                V_random_frobenius_norm_batch = compute_frobenius_norm_batch(V_random_coulomb_matrices_batch)

                V_random_frobenius_norm = sum(V_random_frobenius_norm_batch)

                print("V_random_frobenius_norm", V_random_frobenius_norm)

                #non diagonalized version for testing
                V_random_coulomb_matrices_batch_non_diag = compute_coulomb_matrices_batch(chain_random_batch_original_fragments.cpu(), diagonalize=False)

                V_random_frobenius_norm_batch_non_diag = compute_frobenius_norm_batch(V_random_coulomb_matrices_batch_non_diag)

                V_random_frobenius_norm_non_diag = sum(V_random_frobenius_norm_batch_non_diag)

                print("V_random_frobenius_norm_non_diag", V_random_frobenius_norm_non_diag)
                
                # print("V_random_frobenius_norm", V_random_frobenius_norm)
                

                for r_frob in V_random_frobenius_norm_batch:
                    frobenius_norm_random_samples.append(r_frob)
                
                

                marginal_contrib_frobenius_norm += (V_j_plus_frobenius_norm - V_j_minus_frobenius_norm)

                

            phi_atoms[fragment_indices[j].item()] = [0]    
            phi_atoms[fragment_indices[j].item()][0] = marginal_contrib_frobenius_norm/M #j is the index of the fragment atom in the fragment indices tensor

        print(data["name"])

        phi_atoms_fronebius_norm = {}
        for atom_index, phi_values in phi_atoms.items():
            phi_atoms_fronebius_norm[atom_index] = phi_values[0]
            
            # phi_atoms_hausdorff[atom_index] = phi_values[2]

        
        # Save phi_atoms to a text file
        with open(f'{folder_save_path}/phi_atoms_{data_index}.txt', 'w') as write_file:
            write_file.write("sample name: " + str(data["name"]) + "\n")
            write_file.write("atom_index,shapley_value\n")
            for atom_index, phi_values in phi_atoms.items():
                write_file.write(f"{atom_index},{phi_values[0]}\n")

            write_file.write("\n")
            # save sum of phi values for disance and cosine similarity
            sum_shapley_values = sum([p_values[0] for p_values in phi_atoms.values()])
            write_file.write("Sum of Shapley values:")
            write_file.write(str(sum_shapley_values.item()) + "\n")
            
            # write_file.write("Sum of phi values for hausdorff\n")
            # write_file.write(str(sum([p_values[2] for p_values in phi_atoms.values()])) + "\n")     
            
            # write_file.write("Average hausdorff distance random samples:\n")
            # write_file.write(str(sum(hausdorff_distances_random_samples)/len(hausdorff_distances_random_samples)) + "\n")      
            
            # write_file.write("Hausdorff distances random samples\n")
            # write_file.write(str(hausdorff_distances_random_samples) + "\n")

            write_file.write("Frobenius norm of original molecule:")
            write_file.write(str(frobenius_norm_original_linker.item()) + "\n")

            average_frobenius_norm_random_samples = sum(frobenius_norm_random_samples)/len(frobenius_norm_random_samples)

            write_file.write("Average Frobenius norm of random samples:")
            write_file.write(str(average_frobenius_norm_random_samples.item()) + "\n")

            
            ideal_sum_shapley_values = frobenius_norm_original_linker - average_frobenius_norm_random_samples

            approx_error = sum_shapley_values - ideal_sum_shapley_values
            abs_approx_error = abs(approx_error)
            write_file.write("Approximation error:")
            write_file.write(str(approx_error.item()) + "\n")
            write_file.write("Absolute approximation error:")
            write_file.write(str(abs_approx_error.item()) + "\n")
            
            write_file.write("Frobenius norm of random samples:\n")
            write_file.write(str(frobenius_norm_random_samples) + "\n")

        if SAVE_VISUALIZATION:
            phi_values_for_viz = phi_atoms_fronebius_norm

            # Saving chains and final states
            for i in range(len(data['positions'])):
                chain = chain_batch[:, i, :, :]
                assert chain.shape[0] == keep_frames
                assert chain.shape[1] == data['positions'].shape[1]
                assert chain.shape[2] == data['positions'].shape[2] + data['one_hot'].shape[2] + model.include_charges

                # Saving chains
                name = str(i + start)
                chain_output = os.path.join(chains_output_dir, name)
                os.makedirs(chain_output, exist_ok=True)
                
                #save initial random distrubution with noise
                positions_combined = torch.zeros_like(data['positions'])
                one_hot_combined = torch.zeros_like(data['one_hot'])

                # Iterate over each atom and decide whether to use original or noisy data
                for atom_idx in range(data['positions'].shape[1]):
                    if data['fragment_mask'][0, atom_idx] == 1:
                        # Use original positions and features for fragment atoms
                        positions_combined[:, atom_idx, :] = data['positions'][:, atom_idx, :]
                        one_hot_combined[:, atom_idx, :] = data['one_hot'][:, atom_idx, :]
                        # atom_mask_combined[:, atom_idx] = data['atom_mask'][:, atom_idx]
                    else:
                        # Use noisy positions and features for linker atoms
                        positions_combined[:, atom_idx, :] = noisy_positions_present_atoms[:, atom_idx, :]
                        one_hot_combined[:, atom_idx, :] = noisy_features_present_atoms[:, atom_idx, :]

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
                one_hot = chain[:, :, 3:] #@mastro, added last atom type (not sure whyt it was not included...) However, TODO check again -> is it the atomic_number? But checking dimensions it did not look like it. Anyway, this should have no effect since the charge/atomic_number is always 0 in our case
                positions = chain[:, :, :3]
                chain_node_mask = torch.cat([node_mask[i].unsqueeze(0) for _ in range(keep_frames)], dim=0)
                names = [f'{name}_{j}' for j in range(keep_frames)]

                save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=model.is_geom)
                invert_colormap = False
                if average_frobenius_norm_random_samples > frobenius_norm_original_linker:
                    invert_colormap = True

                visualize_chain_xai(
                    chain_output,
                    spheres_3d=False,
                    alpha=0.7,
                    bg='white',
                    is_geom=model.is_geom,
                    fragment_mask=data['fragment_mask'][i].squeeze(),
                    phi_values=phi_values_for_viz,
                    invert_colormap=invert_colormap
                )

                # Saving final prediction and ground truth separately
                true_one_hot = data['one_hot'][i].unsqueeze(0)
                true_positions = data['positions'][i].unsqueeze(0)
                true_node_mask = data['atom_mask'][i].unsqueeze(0)
                save_xyz_file(
                    final_states_output_dir,
                    true_one_hot,
                    true_positions,
                    true_node_mask,
                    names=[f'{name}_true'],
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
                    names=[f'{name}_pred'],
                    is_geom=model.is_geom
                )

            start += len(data['positions'])

        

