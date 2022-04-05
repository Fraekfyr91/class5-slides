def estimate( 
        y: np.array, x: np.array, transform='', t: int=None, robust_se = True
    ) -> list:
    """Uses the provided estimator (mostly OLS for now, and therefore we do 
    not need to provide the estimator) to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.array): Dependent variable (Needs to have shape 2D shape)
        >> x (np.array): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >> t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, t)
    if robust_se:
        cov, se = robust(x, residual, t)
    t_values = b_hat/se
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.array, x: np.array) -> np.array:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.array): Dependent variable (Needs to have shape 2D shape)
        >> x (np.array): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.array, 
        t: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.array): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    k = x.shape[1]
    if transform in ('', 'fd', 'be'):
        n = x.shape[0]
    else:
        n = x.shape[0]/t

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(n - k)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(n * (t - 1) - k))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(t * n - k))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se


def robust( x: np.array, residual: np.array, t:int) -> tuple:
    # If only cross sectinoal, we can easiily use the diagonal.
    if not t:
        uhat2 = residual * residual
        diag = np.diag(uhat2.reshape(-1, ))
        cov = la.inv(x.T@x) @ (x.T@diag@x) @ la.inv(x.T@x)
    
    # Else we loop over each individual.
    else:
        n = int(residual.size / t)
        k = x.shape[1]
        diag = np.zeros((k, k))
        for i in range(0, n*t, t):
            slice_obj = slice(i, i + t)
            uhat2 = residual[slice_obj]@residual[slice_obj].T
            diag += x[slice_obj].T @ uhat2 @ x[slice_obj]
        cov = la.inv(x.T@x)@(diag)@la.inv(x.T@x)
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se