def plotPCA(adata_dict, hvg_range=(2000,8500,500)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    N_hvg_list=list(range(hvg_range))
    for cl, adata in adata_dict.items():
        print(f'Cell line: {cl}')
        #print(f'Number of cell lines:{len(adata.obs["CellLineName"].unique())}')
        fig, axes = plt.subplots(math.ceil(len(N_HVG)/4), 4, figsize=(10, 10))
        axes = axes.flatten()
        for idx, N in enumerate(range(*hvg_range)):
            pcomp = adata.obsm[f"PCA_n{N}_HVG"][:, :2]
            df = pd.DataFrame(pcomp, columns=["PC1", "PC2"], index=adata.obs.index)
            df["CellLine"] = adata.obs["CellLineName"]
            sns.scatterplot(x="PC1", y="PC2", data=df, hue="CellLine", ax=axes[idx], legend=False, s=10, alpha=0.5)
            axes[idx].set_xlabel("PC1",fontsize=10)
            axes[idx].set_ylabel("PC2",fontsize=10)
            axes[idx].tick_params(axis="x", labelsize=10)
            axes[idx].tick_params(axis="y", labelsize=10)
            axes[idx].set_title(f"PCA {N} HVG", fontsize=10)
        for idx in range(len(N_hvg_list), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.show()


