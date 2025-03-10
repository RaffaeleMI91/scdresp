def mad_thresholds(data, nmads):
    """
    Calculate thresholds based on the Median Absolute Deviation (MAD).
    
    Parameters:
    - data: array-like, the input data for which thresholds will be calculated.
    - nmads: float, the number of MADs to define the upper and lower limits.
    
    Returns:
    - dict with median, upper limit, and lower limit.
    """
    import numpy as np
    import astropy.stats as ap
    
    # Calculate median and MAD-based thresholds
    m = np.median(data)
    mad = ap.median_absolute_deviation(data)
    upper = m + nmads * mad
    lower = m - nmads * mad
    
    # Return rounded thresholds
    return {"median": np.round(m, 3), "upper_limit": np.round(upper, 3), "lower_limit": np.round(lower, 3)}


def mad_filter(adata, feature, nmads):
    """
    Identify outliers in a feature based on MAD thresholds.

    Parameters:
    - adata: AnnData object, contains the dataset.
    - feature: str, the name of the feature in adata.obs to analyze.
    - nmads: float, the number of MADs to define outlier thresholds.

    Returns:
    - pd.Series (bool) indicating whether each observation is an outlier.
    """
    import numpy as np

    # Extract the feature distribution
    f_distr = adata.obs[feature]

    # Compute MAD thresholds
    th = mad_thresholds(f_distr, nmads)

    # Identify outliers
    outlier = (f_distr < th["lower_limit"]) | (f_distr > th["upper_limit"])

    return outlier
