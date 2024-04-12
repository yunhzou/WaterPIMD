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
    plt.xlabel("Steps")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.title("Potential Energy vs Time step")
    if show:
        plt.show()


def full_autocorrelation(series, show: bool = False):
    """
    Compute the autocorrelation of the specified series for all possible lags.
    This should be the correct one 

    :param series: The time series data.
    :return: A numpy array containing the autocorrelation values for all lags.
    """
    #series = np.mean(series, axis=1)
    N = len(series)
    # Subtract the mean from the series
    series_mean_subtracted = series - np.mean(series)
    
    # Calculate the autocorrelation using the numpy correlate function
    autocorrelations = np.correlate(series_mean_subtracted, series_mean_subtracted, mode='full')[N-1:] / N
    plt.plot(autocorrelations)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function of Potential Energy")
    if show:
        plt.show()
    return autocorrelations

def find_nearest_half_life(correlation_array):
    """
    Find the index of the point in the correlation array that is closest to C(0)/e,
    where C(0) is the first element of the correlation array and e is the base of the
    natural logarithm.

    :param correlation_array: The 1D array representing the autocorrelation function.
    :return: The index of the closest point to C(0)/e.
    """
    # Calculate the target value as C(0)/e
    target_value = correlation_array[0] / np.e
    
    # Find the index of the point closest to the target value
    # We use np.abs to find the absolute difference and argmin to find the index of the minimum value
    index = (np.abs(correlation_array - target_value)).argmin()
    
    return index

def calculate_correlation_time(series):
    """
    Compute the correlation time for the given time series data.
    
    :param series: The time series data.
    :return: The correlation time, tau_c.
    """
    # Subtract the mean from the series to get δA(t)
    series = np.mean(series, axis=1)
    
    # Calculate the full autocorrelation function
    autocorrelations = full_autocorrelation(series, show = True)
    
    # Sum the autocorrelation function values to estimate the integral
    tau_c = find_nearest_half_life(autocorrelations)
    
    return tau_c

if __name__ == "__main__":
    result_dir = r"Result\archive1"
    pe = np.load(os.path.join(result_dir, "pe.npy"))
    plt.plot(pe.mean(axis=1))
    plt.show()
    #print(calculate_correlation_time(pe))

