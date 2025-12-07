# basic_datasets.py
# Gerry Angelatos

# Below we define functions to generate data for a pair of basic classification tasks:
# n-dimensional radius (n-spheres) and 2-d spirals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Circles or N-spheres Dataset
def nsphere_sample(Ns, ndim=2):
    """
    Draw Ns random samples uniformly distributed on the surface of a unit n-sphere
    multiply by R to have samples on surface of radius R
    Parameters
    -----------
    Ns
        number of random points on surface of n-sphere
    ndim
        dimension of nsphere
    Returns
    ----------
    np.array
       (ndim x Ns) array of uniform samples on sphere surface

    """
    vec = np.random.randn(ndim, Ns)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def Noisy_nsphere_sample(dr, Ns, ndim=2):
    """
    Draw Ns random samples on the surface of n-spheres with radii 
    uniformly distributed within dr
    multiply by R to have samples between R-dr*R <r < R+dr*R
    Parameters
    -----------
    dr
        relative variation in radius 1-dr <r< 1+dr
    Ns
        number of random points on surface of n-sphere
    ndim
        dimension of nsphere
    Returns
    ----------
    np.array
       (ndim x Ns) array of uniform samples over spheres 
       with radii uniform distributed 1-dr <r< 1+dr

    """
    return nsphere_sample(int(Ns), ndim) * (1+dr-2*dr*np.random.rand(int(Ns)))


# spirals dataset

def Spiral_sample2(dW, Ns, ts = 0, Nturns = 3, Sep = 0.05, seed=100):
    """
    Generates sample of a 2d spiral with N turns and standard deviation dw
    This version '2' adds noise normal to the spiral line (ie radial white noise)
    and normalizes the dataset relative so the noise free set is in [-1, 1]x2
    Parameters
    -----------
    dW
        width of spiral arms (noise strength),
        scales with the number of turns, so strength of 1
        gives about ~90% probability of falling in the correct arm
    Ns
        number of uniformly distributed points in spiral
    ts
        starting point (radians) relative to y-axis
    Nturns
        number of turns (radians/pi)
    Sep
        starting displacement from origin
    seed
        random seed so dataset is repeateable
    Returns
    ----------
    np.array
       (2 x Ns) array of uniform samples over spiral length,
       normally distributed at specific angle
    """
    rs = np.random.RandomState(seed = seed + int(ts*100) )
    theta = np.sqrt(rs.rand(Ns)) * Nturns * np.pi
    r = theta + np.pi*Sep
    spiral = np.array([(np.cos(ts)*np.cos(theta) - np.sin(ts)*np.sin(theta)) *r,
                      (np.sin(ts)*np.cos(theta) + np.cos(ts)*np.sin(theta) ) *r])
    Norm =  np.array([spiral[0]-spiral[1]/r, spiral[1]-spiral[0]/r]) # normal vector to each point
    return (spiral + (dW * Norm / np.sqrt(Norm[0]**2 + Norm[1]**2))* rs.randn(1, Ns) )/np.max(np.abs(spiral)) # add noise, normalize


def Spiral_line(Ns, ts = 0, Nturns = 3, Sep = 0.05):
    """
    Generates a 2d spiral with N turns 
    Parameters
    -----------
    dW
        width of spiral arms (noise strength)
    Ns
        number of equally spaced points in spiral
    ts
        starting point (radians) relative to y-axis
    Nturns
        number of turns (radians/pi)
    Sep
        starting displacement from origin
    Returns
    ----------
    np.array
       (2 x Ns) array tracing out a spiral in the x-y plane from the origin

    """
    theta = np.linspace(0, 1, Ns) * Nturns * np.pi
    r = theta + np.pi*Sep
    spiral = np.array([(np.cos(ts)*np.cos(theta) - np.sin(ts)*np.sin(theta)) *r,
                      (np.sin(ts)*np.cos(theta) + np.cos(ts)*np.sin(theta) ) *r])
    return spiral 



def Spiral_sample(dW, Ns, ts = 0, Nturns = 3, Sep = 0.05, seed=100):
    """
    Generates sample of a 2d spiral with N turns and standard deviation dw
    Parameters
    -----------
    dW
        width of spiral arms (noise strength)
    Ns
        number of uniformly distributed points in spiral
    ts
        starting point (radians) relative to y-axis
    Nturns
        number of turns (radians/pi)
    Sep
        starting displacement from origin
    seed
        random seed so dataset is repeateable
    Returns
    ----------
    np.array
       (2 x Ns) array of uniform samples over spiral length,
       normally distributed at specific angle

    
    """
    rs = np.random.RandomState(seed = seed + int(ts*100) )
    theta = np.sqrt(rs.rand(Ns)) * Nturns * np.pi
    r = theta + np.pi*Sep
    spiral = np.array([(np.cos(ts)*np.cos(theta) - np.sin(ts)*np.sin(theta)) *r,
                      (np.sin(ts)*np.cos(theta) + np.cos(ts)*np.sin(theta) ) *r])
    return spiral + rs.randn(2, Ns) * dW


def get_bas_example():
    """Return the exact BAS dataset and labels from the user's example."""
    BAS_data = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    x_BAS = np.array(BAS_data)
    (N_BAS, D_BAS) = x_BAS.shape
    xstr_BAS = [str(key) for key in x_BAS]
    xstr_BAS_binary = [ind for ind in range(len(xstr_BAS))]
    Y_BAS = np.array([0, 0, 0, 1,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     1, 0, 0, 0])
    BAS_images = [np.array(BAS_data[q]).reshape(2,2) for q in range(N_BAS)]
    return x_BAS, Y_BAS, BAS_images, xstr_BAS, xstr_BAS_binary


def get_noisy_bas_example(n=2, num_samples=100, noise_level=0.2, seed=42):
    """Generate a noisy BAS dataset"""
    rng = np.random.default_rng(seed)
    x_BAS = rng.integers(0, 2, size=(num_samples, n*n))
    # Add bit-flip noise
    noise_mask = rng.random((num_samples, n*n)) < noise_level
    x_BAS_noisy = np.bitwise_xor(x_BAS, noise_mask.astype(int))
    # Example label: parity (sum mod 2)
    Y_BAS = np.sum(x_BAS_noisy, axis=1) % 2
    BAS_images = [x_BAS_noisy[i].reshape(n, n) for i in range(num_samples)]
    xstr_BAS = [str(img) for img in x_BAS_noisy]
    xstr_BAS_binary = list(range(num_samples))
    return x_BAS_noisy, Y_BAS, BAS_images, xstr_BAS, xstr_BAS_binary


def plot_bas_example(BAS_images, Y_BAS, cutfact=2, figsize=(16,6)):
    """Plot the BAS example dataset in a grid."""
    N_BAS = len(BAS_images)
    fig_BAS, ax_BAS = plt.subplots(cutfact, int(N_BAS/cutfact), figsize=figsize)
    ax_BAS = ax_BAS.flatten()
    for q in range(N_BAS):
        ax_BAS[q].matshow(BAS_images[q], cmap='binary', alpha=1, clim=[0,1])
        ax_BAS[q].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax_BAS[q].set_title(f"Input {q}, target {Y_BAS[q]}")
    plt.tight_layout()
    return fig_BAS, ax_BAS



