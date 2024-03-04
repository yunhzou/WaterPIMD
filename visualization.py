import matplotlib.pyplot as plt
import numpy as np
import os

def plot_PE(potential_arr: np.ndarray, show: bool = True):
    """
    plot potential energy vs time step

    Args:
        potential_arr: potential energy array of shape (simulation_steps,P)
    """
    # average over beads to shape (simulation_steps,)
    pe = np.mean(potential_arr, axis=1)/4.184
    plt.plot(pe)
    plt.xlabel("Time step")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.title("Potential Energy vs Time step")
    if show:
        plt.show()

def autocorrelation(x):
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size//2] / np.sum(xp**2)

def plot_autocorrelation(potential_arr, show: bool = True):
    """
    Plot the autocorrelation function of the potential energy array.

    Args:
        potential_arr: potential energy array of shape (simulation_steps, P)
    """
    # average over beads to shape (simulation_steps,)
    pe = np.mean(potential_arr, axis=1)
    
    # normalize the potential energy
    pe_normalized = pe / np.max(np.abs(pe))
    
    # calculate the autocorrelation function
    acf = autocorrelation(pe_normalized)
    
    # plot the autocorrelation function
    plt.plot(acf)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function of Potential Energy")
    if show:
        plt.show()

if __name__ == "__main__":
    result_dir = r"Result"
    pe = np.load(os.path.join(result_dir, "pe.npy"))
    #plot_PE(pe, show = False)
    plot_autocorrelation(pe[:200],show=False)
    plot_autocorrelation(pe)

