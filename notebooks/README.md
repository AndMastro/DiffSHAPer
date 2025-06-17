# Notebooks overview

ℹ️ This folder contains Jupyter notebooks that can be used to execute the analyses performed in the paper, as follows:

- 📈 **shapley_value_variance_study.ipynb**  
    This notebook contains code for the study of the variance of the Shapley value approximation.

- 📊 **shapley_value_distribution_analysis.ipynb**  
    Analysis of the distribution of the Shapley values obtained starting from different initial states as well as statistical significance analyses.

- 🧬 **diffusion_explainer_coulomb_matrix_control.ipynb**  
    Control experiments using a molecular mechanics Coulomb-matrix-based value function for the Shapley values.

- ⚓ **e3_transformation_control.ipynb**  
    Analysis of anchor and non-anchor atoms' Shapley values in presence of E(3)-transformations.

- 🌟 **neighbor_atom_analysis.ipynb**  
    Analysis of the impact of important non-anchor neighboring atoms. For a fine-grain analysis, the parameter ``KEEP_FRAMES`` in ``config.yml`` should be set at least to 30. In order to perform atom injection, the corresponing parameter in the configuration files should be set to ``True``.

- 📏 **hausdorff_distace_plotter.ipynb**  
    After having computed the distances using the previous notebook, this code plots the distances when atoms are removed and injected back into the molecule.


> ⚠️ **Note:** Some fluctuations in the results may be expected due to nondeterministic elements involved that cannot be fully controlled.