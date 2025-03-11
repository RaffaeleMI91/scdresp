def mad_thresholds(data, nmads):
  
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

    import numpy as np

    # Extract the feature distribution
    f_distr = adata.obs[feature]
    # Compute MAD thresholds
    th = mad_thresholds(f_distr, nmads)
    # Identify outliers
    outliers = (f_distr < th["lower_limit"]) | (f_distr > th["upper_limit"])

    return outliers
