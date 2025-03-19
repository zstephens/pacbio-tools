import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as mpl
from sklearn.metrics import r2_score


def bimodal_pdf(x, A, B):
    # component 1: N(x, 0.5 - A, B)
    pdf1 = stats.norm.pdf(x, loc=0.5-A, scale=np.sqrt(B))
    # component 2: N(x, 0.5 + A, B)
    pdf2 = stats.norm.pdf(x, loc=0.5+A, scale=np.sqrt(B))
    # mixture with equal weights
    return 0.5 * pdf1 + 0.5 * pdf2


def negative_log_likelihood(params, data):
    A, B = params

    # Ensure B is positive (variance must be positive)
    if B <= 0:
        return np.inf

    # Compute PDF for each data point
    pdf_values = bimodal_pdf(data, A, B)

    # Avoid numerical issues with log(0)
    pdf_values = np.maximum(pdf_values, 1e-10)

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(pdf_values))

    return -log_likelihood  # Return negative for minimization


def fit_bimodal_gaussian(data, initial_guess=None):

    if initial_guess is None:
        # Heuristic for initial guess:
        # A: half the distance between modes (try 1/4 of data range)
        # B: variance of the data divided by 2 (since we have two components)
        data_range = np.max(data) - np.min(data)
        initial_A = data_range / 4
        initial_B = np.var(data) / 2
        initial_guess = [initial_A, initial_B]

    # Bounds for parameters (A can be any value, B must be positive)
    bounds = [(None, None), (1e-6, None)]  # A: unbounded, B: positive

    # Perform optimization
    result = minimize(
        negative_log_likelihood,
        initial_guess,
        args=(data,),
        bounds=bounds,
        method='L-BFGS-B'
    )
    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    A_est, B_est = result.x

    # Calculate the maximized log-likelihood (negate the minimized negative log-likelihood)
    max_log_likelihood = -result.fun

    # Calculate AIC: -2*log(L) + 2k, where k is the number of parameters (here k=2)
    aic = -2 * max_log_likelihood + 2 * 2

    # Calculate BIC: -2*log(L) + k*log(n), where n is the number of data points
    bic = -2 * max_log_likelihood + 2 * np.log(len(data))

    # Calculate individual component likelihoods for comparison
    component1_ll = np.sum(np.log(stats.norm.pdf(data, loc=0.5-A_est, scale=np.sqrt(B_est))))
    component2_ll = np.sum(np.log(stats.norm.pdf(data, loc=0.5+A_est, scale=np.sqrt(B_est))))

    # Calculate empirical PDF using histogram
    hist_counts, hist_bins = np.histogram(data, bins=30, density=True)
    bin_centers = 0.5 * (hist_bins[1:] + hist_bins[:-1])

    # Calculate model PDF at bin centers
    model_pdf = bimodal_pdf(bin_centers, A_est, B_est)

    # Calculate R-squared between empirical and model PDFs
    r_squared = r2_score(hist_counts, model_pdf)

    # compare against single gaussian fit
    single_gaussian_params = {'loc': np.mean(data), 'scale': np.std(data)}
    single_gaussian_ll = np.sum(np.log(stats.norm.pdf(data, **single_gaussian_params)))
    lr_stat = 2 * (max_log_likelihood - single_gaussian_ll)

    return {
        'A': A_est,
        'B': B_est,
        'success': result.success,
        'max_log_likelihood': max_log_likelihood,
        'aic': aic,
        'bic': bic,
        'component1_log_likelihood': component1_ll,
        'component2_log_likelihood': component2_ll,
        'r_squared': r_squared,
        'neg_log_likelihood': result.fun,
        'single_gaussian_log_likelihood': single_gaussian_ll,
        'single_gaussian_test_statistic': lr_stat,
        'single_gaussian_p-value': 1 - stats.chi2.cdf(lr_stat, df=2)
    }


def plot_bimodal_fit(data, A, B, out_fn, plot_title=None):
    """
    Plot the data histogram and fitted PDF.

    Parameters:
    -----------
    data : array-like
        Observed data points
    A : float
        Estimated A parameter
    B : float
        Estimated B parameter
    metrics : dict, optional
        Dictionary containing goodness-of-fit metrics to display on the plot
    """
    # Create a range of x values for plotting the PDF
    x = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)
    pdf = bimodal_pdf(x, A, B)

    # Plot histogram of data
    fig = mpl.figure(9, figsize=(10, 6))
    hist_counts, hist_bins, _ = mpl.hist(data, bins=30, density=True, alpha=0.6, label='Data')

    # Scale the PDF to match the histogram
    mpl.plot(x, pdf, 'r-', lw=2, label=f'Fitted PDF: A={A:.4f}, B={B:.4f}')

    # Add the individual components
    component1 = stats.norm.pdf(x, loc=0.5-A, scale=np.sqrt(B)) * 0.5
    component2 = stats.norm.pdf(x, loc=0.5+A, scale=np.sqrt(B)) * 0.5
    mpl.plot(x, component1, 'g--', lw=1, label=f'Component 1: μ={0.5-A:.4f}')
    mpl.plot(x, component2, 'b--', lw=1, label=f'Component 2: μ={0.5+A:.4f}')

    if plot_title is not None:
        mpl.title(plot_title)
    else:
        mpl.title('Bimodal Gaussian Fit')

    mpl.xlim([0,1])
    mpl.xlabel('x')
    mpl.ylabel('Density')
    mpl.legend()
    mpl.grid(alpha=0.3)

    mpl.savefig(out_fn)
    mpl.close(fig)


def generate_sample_data(A=0.3, B=0.04, n_samples=1000, seed=42):
    """Generate synthetic data from the bimodal distribution"""
    np.random.seed(seed)

    # Decide which component each sample comes from
    component = np.random.choice([0, 1], size=n_samples)

    # Generate data from the respective components
    data = np.zeros(n_samples)
    for i in range(n_samples):
        if component[i] == 0:
            data[i] = np.random.normal(0.5-A, np.sqrt(B))
        else:
            data[i] = np.random.normal(0.5+A, np.sqrt(B))
    return data


if __name__ == "__main__":
    # Generate sample data with true parameters A=0.3, B=0.04
    data = generate_sample_data(A=0.3, B=0.04, n_samples=1000)

    # Fit the model
    results = fit_bimodal_gaussian(data)

    print(f"True parameters: A=0.3, B=0.04")
    print(f"Estimated parameters: A={results['A']:.4f}, B={results['B']:.4f}")

    # Plot the results with goodness-of-fit metrics
    plot_results(data, results['A'], results['B'], metrics=results)

    # Print additional goodness-of-fit metrics
    print(f"Maximized Log-Likelihood: {results['max_log_likelihood']:.4f}")
    print(f"AIC: {results['aic']:.4f}")
    print(f"BIC: {results['bic']:.4f}")

    # Compare with a single Gaussian fit
    single_gaussian_params = {'loc': np.mean(data), 'scale': np.std(data)}
    single_gaussian_ll = np.sum(np.log(stats.norm.pdf(data, **single_gaussian_params)))
    print(f"\nSingle Gaussian fit:")
    print(f"Log-Likelihood: {single_gaussian_ll:.4f}")
    print(f"AIC: {-2 * single_gaussian_ll + 2 * 2:.4f}")  # 2 parameters: mean and variance
    print(f"BIC: {-2 * single_gaussian_ll + 2 * np.log(len(data)):.4f}")

    # Likelihood ratio test
    lr_stat = 2 * (results['max_log_likelihood'] - single_gaussian_ll)
    print(f"\nLikelihood Ratio Test (bimodal vs. single Gaussian):")
    print(f"Test Statistic: {lr_stat:.4f}")
    print(f"p-value: {1 - stats.chi2.cdf(lr_stat, df=2):.6f}")  # df = difference in parameters

    # To fit your own data, you would do:
    # my_data = np.array([...])  # your data here
    # results = fit_bimodal_gaussian(my_data)
    # plot_results(my_data, results['A'], results['B'], metrics=results)
