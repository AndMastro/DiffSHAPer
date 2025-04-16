# Copyright (c) 2024, Andrea Mastropietro. All rights reserved. This code is
# licensed under the MIT License. See the LICENSE file in the project root for
# more information.

# Standard library imports
import os

# Third-party library imports
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from sklearn.decomposition import PCA

# Project-specific imports
from src.difflinker import const
from src.difflinker.molecule_builder import get_bond_order

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

def compute_hausdorff_distance_batch(mol1, mol2, mask1 = None, mask2 = None):
    """
    Compute the Hausdorff distance between two molecules based on their atomic
    positions and optional masks.

    Args:
        mol1 (torch.Tensor): The first molecule's atomic positions with shape 
            (batch_size, num_atoms, 3).
        mol2 (torch.Tensor): The second molecule's atomic positions with shape 
            (batch_size, num_atoms, 3).
        mask1 (torch.Tensor, optional): A boolean mask indicating which atoms 
            to consider for mol1. If not provided, all atoms will be considered.
        mask2 (torch.Tensor, optional): A boolean mask indicating which atoms 
            to consider for mol2. If not provided, all atoms will be considered.

    Returns:
        list: A list of Hausdorff distances for each pair of molecules in the
        batch.
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

def arrestomomentum():
    # This function is used to intentionally (and magically 🧙‍♂️) interrupt
    # execution for debugging purposes.
    raise KeyboardInterrupt("Debug interrupt.")


# Visualize generated molecules as molecular graphs #maybe substitute phi_values with actula colors
def visualize_mapping_structure(file_names, generation_folder, shapley_values, fragment_mask, linker_mask, save_folder, colormap = 'coolwarm_r'):
    
    for name in file_names:
    # Load generated molecule positions and atom types
        generated_file = os.path.join(generation_folder, name +"_.xyz")
        positions, one_hot, _ = load_molecule_xyz(generated_file, is_geom=False)
        
        atom_types = torch.argmax(one_hot, dim=1).cpu().numpy()

        # Convert positions and atom types to an RDKit molecule
        mol = Chem.RWMol()
        atom_map = {}

        # Add atoms to the molecule
        for idx, atom_type in enumerate(atom_types):
            atom = Chem.Atom(const.IDX2ATOM[atom_type])
            atom_idx = mol.AddAtom(atom)
            atom_map[idx] = atom_idx

        # Add single bonds based on positions
        for idx1 in range(len(positions)):
            for idx2 in range(idx1 + 1, len(positions)):
                dist = np.linalg.norm(positions[idx1] - positions[idx2])
                bond_order = has_bond(const.IDX2ATOM[atom_types[idx1]], const.IDX2ATOM[atom_types[idx2]], dist)
                
                if bond_order > 0:
                    bond_type = const.BOND_DICT[bond_order]
                    mol.AddBond(atom_map[idx1], atom_map[idx2], bond_type)
                    
                    if bond_type == Chem.rdchem.BondType.AROMATIC:
                        bond = mol.GetBondBetweenAtoms(atom_map[idx1], atom_map[idx2])
                        bond.SetIsAromatic(True)

        # Calculate the gradient colors based on phi values
        phi_values_array = np.array(shapley_values)
        cmap = plt.cm.get_cmap(colormap)  # reversed heatmap for distance-based importance
        norm = plt.Normalize(vmin=min(phi_values_array), vmax=max(phi_values_array))
        highlight_colors = {idx: cmap(norm(phi_values_array[idx])) for idx, mask in enumerate(fragment_mask) if mask == 1}
        
        highlight_atoms = list(highlight_colors.keys())
        highlight_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in highlight_atoms and bond.GetEndAtomIdx() in highlight_atoms:
                highlight_bonds.append(bond.GetIdx())
        
        # Assign bond colors based on the colors of the connected atoms
        bond_colors = {}

        # Convert to RDKit molecule and draw
        mol = mol.GetMol()
        atom_colors = {idx: list(map(float, color[:3])) for idx, color in highlight_colors.items()}  # Convert to lists of floats
        atom_colors = {key: [tuple(value)] for key, value in atom_colors.items()}
        
        # Draw linker bonds and atoms in emerald green
        emerald_green = (0.25, 0.63, 0.38, 0.7)  # RGB for emerald green
 
        # Update atom colors for linker atoms
        for idx, mask in enumerate(linker_mask):
            if mask == 1:
                atom_colors[idx] = [emerald_green[:3]]  # Set emerald green for linker atoms
    
        # Update bond colors for linker bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if linker_mask[begin_idx] == 1 and linker_mask[end_idx] == 1:
                bond_colors[bond.GetIdx()] = [emerald_green]  # Set emerald green for linker bonds
        
        # Use DrawMoleculeWithHighlights for visualization
        atom_radii = {idx: 0.5 for idx in range(len(positions))}  # Default atom radius
        # Set radius 0.3 for linker atoms
        for idx, mask in enumerate(linker_mask):
            if mask == 1:
                atom_radii[idx] = 0.3
        
        drawer = Draw.MolDraw2DCairo(800, 800)
        draw_options = drawer.drawOptions()
        draw_options.useBWAtomPalette()
        draw_options.kekulize = False
        
        drawer.DrawMoleculeWithHighlights(
            mol,
            "",  # Legend
            atom_colors,  # Atom colors (highlight_atom_map)
            bond_colors,  # Bond colors (highlight_bond_map)
            atom_radii,  # Atom radii
            {}  # Bond linewidth multipliers
        )

        drawer.FinishDrawing()
        # Convert the drawing to a PNG image
        png_data = drawer.GetDrawingText()
        os.makedirs("temp", exist_ok=True)
        with open("temp/temp_image.png", "wb") as f:
            f.write(png_data)
        
        img = Image.open("temp/temp_image.png")
        
        # Save the image in the "structure" subfolder
        output_file = os.path.join(save_folder, f"{name}_structure.png")
        img.save(output_file, dpi=(300, 300))

def has_bond(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    """
    Determines whether a bond exists between two atoms based on their distance
    and predefined bond length thresholds.

    Args:
        atom1 (str): The identifier or type of the first atom. atom2 (str): The
        identifier or type of the second atom. distance (float): The distance
        between the two atoms in nanometers. 
                          This value will be converted to picometers internally.
        check_exists (bool, optional): If True, checks whether the bond type 
                                       exists in the predefined bond dictionary.
                                       Defaults to True.
        margins (tuple, optional): A tuple of margin values used to adjust bond 
                                   length thresholds. Defaults to
                                   `const.MARGINS_EDM`.

    Returns:
        int: Returns 1 if a bond exists (single bond), otherwise returns 0 (no
        bond).

    Notes:
        - The function assumes that bond lengths are stored in `const.BONDS_1`
          as a nested dictionary where the first atom maps to a dictionary of
          second atoms and their respective bond lengths.
        - The `margins` parameter is used to fine-tune the bond length
          thresholds for stability.
        - Adapted from original code by Ilia Igashov, Hannes Stärk, Clément
          Vignac (c) 2022.
    """
    distance = 100 * distance  

    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    if distance < const.BONDS_1[atom1][atom2] + margins[0]:
        return 1  # Single
    return 0  # No bond

#graph-based visualizaiton functions modified from original code by Ilia
#Igashov, Hannes Stärk, Clément Vignac (c) 2022 distributed under the MIT
#license

def draw_sphere_xai(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) #* 0.8
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, alpha=alpha)

# def plot_molecule_xai(ax, positions, atom_type, alpha, spheres_3d,
#     hex_bg_color, is_geom, fragment_mask=None, phi_values=None): x =
#     positions[:, 0] y = positions[:, 1] z = positions[:, 2] # Hydrogen,
#     Carbon, Nitrogen, Oxygen, Flourine

#     idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

#     colors_dic = np.array(const.COLORS) radius_dic = np.array(const.RADII)
#     area_dic = 1500 * radius_dic ** 2

#     areas = area_dic[atom_type] radii = radius_dic[atom_type] colors =
#     colors_dic[atom_type]

#     if fragment_mask is None: fragment_mask = torch.ones(len(x))

#     for i in range(len(x)): for j in range(i + 1, len(x)): p1 =
#         np.array([x[i], y[i], z[i]]) p2 = np.array([x[j], y[j], z[j]]) dist =
#             np.sqrt(np.sum((p1 - p2) ** 2)) atom1, atom2 =
#             idx2atom[atom_type[i]], idx2atom[atom_type[j]] draw_edge_int =
#             get_bond_order(atom1, atom2, dist) line_width = (3 - 2) * 2 * 2
#             draw_edge = draw_edge_int > 0 if draw_edge: if draw_edge_int == 4:
#             linewidth_factor = 1.5 else: linewidth_factor = 1 linewidth_factor
#             *= 0.5 ax.plot( [x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
#             linewidth=line_width * linewidth_factor * 2, c=hex_bg_color,
#             alpha=alpha )

    

#     if spheres_3d:
        
#         for i, j, k, s, c, f, phi in zip(x, y, z, radii, colors,
#             fragment_mask, phi_values): if f == 1: alpha = 1.0 if phi > 0: c =
#                 'red'

#             draw_sphere_xai(ax, i.item(), j.item(), k.item(), 0.5 * s, c,
#             alpha)

#     else: phi_values_array = np.array(list(phi_values.values()))

#         #draw fragments fragment_mask_on_cpu = fragment_mask.cpu().numpy()
#         colors_fragment = colors[fragment_mask_on_cpu == 1] x_fragment =
#         x[fragment_mask_on_cpu == 1] y_fragment = y[fragment_mask_on_cpu == 1]
#         z_fragment = z[fragment_mask_on_cpu == 1] areas_fragment =
#         areas[fragment_mask_on_cpu == 1]
        
#         # Calculate the gradient colors based on phi values
#         cmap = plt.cm.get_cmap('coolwarm_r') #reversed heatmap for
#         distance-based importance norm =
#         plt.Normalize(vmin=min(phi_values_array), vmax=max(phi_values_array))
#         colors_fragment_shadow = cmap(norm(phi_values_array))
        
#         # ax.scatter(x_fragment, y_fragment, z_fragment, s=areas_fragment, alpha=0.9 * alpha, c=colors_fragment)

#         ax.scatter(x_fragment, y_fragment, z_fragment, s=areas_fragment,
#         alpha=0.9 * alpha, c=colors_fragment,
#         edgecolors=colors_fragment_shadow, linewidths=5, rasterized=False)

#         #draw non-fragment atoms colors = colors[fragment_mask_on_cpu == 0] x
#         = x[fragment_mask_on_cpu == 0] y = y[fragment_mask_on_cpu == 0] z =
#         z[fragment_mask_on_cpu == 0] areas = areas[fragment_mask_on_cpu == 0]
#         ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors,
#         rasterized=False)

def plot_molecule_xai(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, is_geom, fragment_mask=None, phi_values=None, colors_fragment_shadow=None, draw_atom_indices = None):
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

                # Check if at least one atom is a linker atom
                if fragment_mask[i] == 0 or fragment_mask[j] == 0:
                    edge_color = 'gray'
                    current_alpha = 0.5 * alpha
                else:
                    edge_color = hex_bg_color
                    current_alpha = 0.7 * alpha
                
                ax.plot(
                    [x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                    linewidth=line_width * linewidth_factor * 2,
                    c=edge_color,
                    alpha=current_alpha
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
        # ax.scatter(x_fragment, y_fragment, z_fragment, s=areas_fragment,
        # alpha=0.9 * alpha, c=colors_fragment)

        ax.scatter(x_fragment, y_fragment, z_fragment, s=areas_fragment, alpha=0.9 * alpha, c=colors_fragment_shadow, edgecolors=colors_fragment_shadow, linewidths=2.5, rasterized=False)
        
        if draw_atom_indices == "original":
            #get fragment indices using fragment mask
            fragment_indices = np.where(fragment_mask_on_cpu == 1)[0]
            for i, txt in enumerate(fragment_indices):
                ax.text(x_fragment[i], y_fragment[i], z_fragment[i], str(txt), color='black', fontsize=15)
        
        elif draw_atom_indices is None:
            pass

        else:
            for i, txt in enumerate(draw_atom_indices[0]):
                ax.text(x_fragment[i], y_fragment[i], z_fragment[i], str(txt), color='black', fontsize=15)

        

        #draw non-fragment atoms
        colors = np.array([0.25, 0.63, 0.38, 1]) 
        x = x[fragment_mask_on_cpu == 0]
        y = y[fragment_mask_on_cpu == 0]
        z = z[fragment_mask_on_cpu == 0]
        areas = areas[fragment_mask_on_cpu == 0]
        ax.scatter(x, y, z, s=areas, alpha=0.5 * alpha, color=colors, rasterized=False)

        if draw_atom_indices == "original":
            #get non-fragment indices using fragment mask
            non_fragment_indices = np.where(fragment_mask_on_cpu == 0)[0]
            for i, txt in enumerate(non_fragment_indices):
                ax.text(x[i], y[i], z[i], str(txt), color='black', fontsize=15)
        elif draw_atom_indices is None:
            pass
        else:
            for i, txt in enumerate(draw_atom_indices[1]):
                ax.text(x[i], y[i], z[i], str(txt), color='black', fontsize=15)

def plot_data3d_xai(positions, atom_type, is_geom, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1., fragment_mask=None, phi_values=None):
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
        ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, is_geom=is_geom, fragment_mask=fragment_mask, phi_values=phi_values
    )

    max_value = positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    dpi = 300 if spheres_3d else 300 #it was 120 and 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi,
        # transparent=True)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()

def visualize_mapping_graph(
        path, spheres_3d=False, bg="black", alpha=1.0, is_geom=False, fragment_mask=None, phi_values=None, colors_fragment_shadow=None,
                    draw_atom_indices=None
):
    files = load_xyz_files(path)
    save_paths = []

    # Fit PCA to the final molecule – to obtain the best orientation for
    # visualization
    positions, one_hot, charges = load_molecule_xyz(files[-1], is_geom=is_geom)
    pca = PCA(n_components=3)
    pca.fit(positions)

    for i in range(len(files)):
        file = files[i]

        positions, one_hot, charges = load_molecule_xyz(file, is_geom=is_geom)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        # Transform positions of each frame according to the best orientation of
        # the last frame
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
            phi_values=phi_values
        )
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    imageio.mimsave(gif_path, imgs, subrectangles=True)


def save_xyz_file(path, one_hot, positions, node_mask, names, is_geom, suffix=''):
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        n_atoms = mask.sum()
        atom_idx = torch.where(mask)[0]

        f = open(os.path.join(path, f'{names[batch_i]}_{suffix}.xyz'), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in atom_idx:
            atom = atoms[atom_i].item()
            atom = idx2atom[atom]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]
            ))
        f.close()

#@mastro edited
def load_xyz_files(path, suffix='', file_indices=None):
    files = []
    
    for fname in os.listdir(path):
        if fname.endswith(f'_{suffix}.xyz'):
            files.append(fname)
    files = sorted(files, key=lambda f: -int(f.replace(f'_{suffix}.xyz', '').split('_')[-1]))
    
    if file_indices is not None:
        files = [file for file in files if int(file.split('_')[-2]) in file_indices]
    return [os.path.join(path, fname) for fname in files]
    


def load_molecule_xyz(file, is_geom):
    atom2idx = const.GEOM_ATOM2IDX if is_geom else const.ATOM2IDX
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(idx2atom))
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, atom2idx[atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot, charges