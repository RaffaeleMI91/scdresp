def computeHVGs(adata_dict:dict, hvg_range:tuple[int,int,int], remove_mt_rb:bool=True, verbose:bool=True) -> dict:
    
    """
    Compute Highly Variable Genes (HVGs) for each AnnData object in 'adata_dict'
    Parameterts: 
    - adata_dict: dict of AnnData objects
    - hvg_range: tuple (start, stop, range) defining the range of HVGs to compute
    - verbose: bool, wheter to print progress messages

    Returns:
    - Updated 'adata_dict' with HVG results in '.var'
    """
    
    for CL, adata in adata_dict.items():
        if verbose:
            print(f"Computing HVG for Sanger Model ID: {CL}")
        for N in range(*hvg_range):
            sc.pp.highly_variable_genes(adata, n_top_genes=N)
            hvg_col_name = f"highly_variable_n{N}"
            adata.var[hvg_col_name] = adata.var["highly_variable"]
            # Ensure HVG genes are NOT mitochondrial or ribosomal
            if remove_mt_rb:
                if "mt" in adata.var.columns and "ribo" in adata.var.columns:
                    adata.var[hvg_col_name] = np.logical_and(
                        np.logical_xor(
                            np.logical_or(adata.var["mt"], adata.var["ribo"]),
                            adata.var[hvg_col_name]
                        ),
                            adata.var[hvg_col_name]
                    )
    return adata_dict

def computePCA(adata_dict:dict, hvg_range:tuple[int, int, int], verbose:bool=True) -> dict:
    
    """
    Compute Principal Components (PCs) for each AnnData object in 'adata_dict'

    Parameters:
    - adata_dict: dict of AnnData objects
    - hvg_range: tuple (start, stop, step) defining the range of HVGs to compute
    - verbose: bool, whether to print progress messages

    Returns:
    - Updated 'adata_dict' with PCA results in '.obsm'
    """
    
    for CL, adata in adata_dict.items(): 
        if verbose:
            print(f'Computing PCA for Sanger Model ID: {CL}')
        
        expected_hvg_cols = {f"highly_variable_n{N}" for N in range(*hvg_range)}
        observed_hvg_cols = set(adata.var.columns[adata.var.columns.str.startswith("highly_variable_n")])
        
        if not expected_hvg_cols.issubset(observed_hvg_cols):
            raise ValueError(f"Missing HVGs in {CL} dataset. Please, compute top N high variable genes for a range of values first")
            continue

        for N in range(*hvg_range):
            hvg_col_name = f"highly_variable_n{N}"
            if hvg_col_name in adata.var.columns:
                adata_subset = adata[:, adata.var[hvg_col_name]].copy()
                sc.pp.scale(adata_subset, max_value=10)
                sc.pp.pca(adata_subset, svd_solver='arpack')
                adata.obsm[f"PCA_n{N}_HVG"] = adata_subset.obsm["X_pca"]
            else:
                warnings.warn(f"Can't compute PCA for {CL} using {N} HVGs. Ensure the top {N} high variable genes are stored in '.var' before proceeding!", UserWarning)

    return adata_dict

def doKmClustering(adata_dict:dict, hvg_range:tuple[int,int,int], n_pc:int=30, cl_range:tuple[int,int], verbose:bool=True) -> dict:
    
    """
    Perform K-Means clustering on PCA-reduced gene expression data for each AnnData object in `adata_dict`.

    Parameters:
    - adata_dict (dict): Dictionary containing AnnData objects.
    - hvg_range (tuple): (start, stop, step) range for the number of highly variable genes (HVGs).
    - n_pc (int): Number of principal components (PCs) to use for clustering. Default is 30.
    - cl_range (tuple): (min_clusters, max_clusters) range for K-Means clustering.
    - verbose (bool): Whether to print progress messages. Default is True.

    Returns:
    - adata_dict (dict): Updated AnnData dictionary with K-Means cluster assignments in `.obs`.
    """
    
    for CL, adata in adata_dict.items():
        if verbose:
            print(f'Computing k-means clustering for Sanger Model ID: {CL}')
        
        expected_pca_cols = {f"PCA_n{N}_HVG" for N in range(*hvg_range)}
        observed_pca_cols = set(adata.obsm.keys())

        if not expected_pca_cols.issubset(observed_pca_cols):
            raise ValueError(f"Missing PCs in {CL} dataset. Please, compute PCA using a range of top N high variable genes first!")
            continue

        for N in range(*hvg_range):
            km_results = {}
            pca_key = f"PCA_n{N}_HVG"
            if pca_key in adata.obsm.keys(): # check nr 2
                adata.obsm["X_pca"] = adata.obsm[pca_key][:, :n_pc]
                pcomp = adata.obsm["X_pca"]
                for cl in range(*cl_range):
                    kmeans = KMeans(n_clusters=cl, random_state=42, n_init=10)
                    km_results[f"kmeans_n{N}_{cl}_clusters"] = kmeans.fit_predict(pcomp).astype(str)     
                
                # expand adata
                cluster_df = pd.DataFrame(km_results, index=adata.obs.index)
                adata.obs = pd.concat([adata.obs, cluster_df], axis=1)
                adata.obs = adata.obs.copy()
            else:
                warnings.warn(f"Cannot perform K-Means clustering for {CL} with {N} HVGs. Ensure PCA, computed on the top {N} HVGs, is available in .obsm before proceeding!", UserWarning)
    
    return(adata_dict)


def doGMixClustering(adata_dict: dict, hvg_range:tuple=(2000, 8500, 500), n_pc:int=30, cl_range:tuple=(2,11)):
    
    import pandas as pd
    import scanpy as sc
    from sklearn.mixture import GaussianMixture

    for idx, (CL, adata) in enumerate(adata_dict.items()):
        print(f'Processing {idx} : {CL}')
        for N in range(*hvg_range):
            gmm_results = {}
            pca_key = f"PCA_n{N}_HVG"
            adata.obsm["X_pca"] = adata.obsm[pca_key][:, :n_pc]
            pcomp=adata.obsm["X_pca"]
            for cl in range(*cl_range):
                gmm=GaussianMixture(n_components=cl, covariance_type='full', random_state=42)
                gmm_results[f"gmm_n{N}_{cl}_clusters"]=gmm.fit_predict(pcomp).astype(str)
            
            # extend adata .obs
            cluster_df=pd.DataFrame(gmm_results, index=adata.obs.index)
            adata.obs=pd.concat([adata.obs, cluster_df], axis=1)
            adata.obs=adata.obs.copy()

def doLeidenClustering(adata_dict, hvg_range=(2000,8500,500), n_knn=15, n_pc=30, res_list=[0.2, 0.4,0.6,0.8,1,1.2]):
    """
    Performs Leiden clustering on PCA-reduced gene expression data.

    Parameters:
    - adata_dict (dict): Dictionary containing AnnData objects.
    - hvg_range (tuple): (start, stop, step) for the number of highly variable genes (HVG) to test.
    - n_knn (int): Number of nearest neighbors for graph construction.
    - n_pc (int): Number of principal components (PCs) to use in clustering.
    - res_list (list): List of resolution values for Leiden clustering.
    
    Returns:
    - Update "adata.obs" with leiden clusters assignments. 
    """
    import anndata
    import scanpy as sc

    for idx, (CL, adata) in enumerate(adata_dict.items()):
        print(f'Processing {idx} : {CL}')
        for N in range(*hvg_range):
            adata_hvg = adata[:, adata.var[f"highly_variable_n{N}"]].copy()
            adata_hvg.obsm["X_pca"] = adata.obsm[f"PCA_n{N}_HVG"]
            sc.pp.neighbors(adata_hvg, n_neighbors=n_knn, n_pcs=n_pc)
            # update adata .obsp and .uns
            adata.obsp[f"connectivities_n{N}_HVG"] = adata_hvg.obsp["connectivities"]
            adata.obsp[f"distances_n{N}_HVG"] = adata_hvg.obsp["distances"]
            adata.uns[f"neighbors_n{N}_HVG"] = adata_hvg.uns["neighbors"]
            for res in res_list:
                sc.tl.leiden(adata_hvg, resolution=res, random_state=2, n_iterations=-1)
                # extend adata .obs
                adata.obs[f"leiden_n{N}_{res}_clusters"] = adata_hvg.obs["leiden"]

def categorizeOTS(data_dict, perc=90, plot_figure=True):
    
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ncols = 3
    nrows = math.ceil(len(data_dict) / ncols)  
    
    if plot_figure:
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 25))
        axes = axes.flatten()
    
    for idx, (drug, data) in enumerate(data_dict.items()):
        df = data["Metadata"]["OncotreeSubtype"]
        threshold = np.percentile(df.value_counts(), perc)
        counts_df = df.value_counts().reset_index()
        counts_df.columns = ["Subtype", "Count"]
        
        counts_df["Class"] = np.where(counts_df["Count"] >= threshold, "over", "down")

        data_dict[drug]["Metadata"] = data["Metadata"].copy()  
        data_dict[drug]["Metadata"]["Class"] = df.map(counts_df.set_index("Subtype")["Class"])

        if plot_figure:
            sns.barplot(x=counts_df["Subtype"], y=counts_df["Count"], hue=counts_df["Class"], ax=axes[idx])
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right", fontsize=12)
            axes[idx].tick_params(axis="y", labelsize=12)
            axes[idx].set_title(f"{drug}; nr of cell lines: {len(data['OBS'])}", fontsize=14)

    if plot_figure:
        for idx in range(len(data_dict), len(axes)):
            axes[idx].set_visible(False)

        plt.subplots_adjust(hspace=0.8, wspace=0.5)
        plt.tight_layout()
        plt.show()
