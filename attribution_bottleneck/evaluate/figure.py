import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def values_with_gaussian(z, bins):

    # Plot histogram
    plt.hist(z.ravel(), bins=bins, weights=np.full_like(z.ravel(), 1 / z.size))

    # Fit normal distribution
    mu, std = z.mean(), z.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)