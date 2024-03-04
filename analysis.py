import numpy as np
import os 
import json


def errorpropagation(data):
    """
    Calculate the standard error of the mean (SEM) for a given dataset.
    
    Parameters:
    - data: numpy.ndarray, a 1D array of data points from which the SEM is calculated.
    
    Returns:
    - float: The calculated standard error of the mean.
    """
    ndim = len(data)
    error = np.std(data, ddof=0) / np.sqrt(ndim)
    return error


def maxError_byBinning(mean, data, workingNdim):
    """
    Calculate the maximum error estimate using the binning method for a dataset and a given mean.
    
    Parameters:
    - mean: float, the mean value of the dataset for which the error is estimated.
    - data: numpy.ndarray, the dataset from which to estimate the error. Should be a 1D array.
    - workingNdim: int, the number of iterations for binning.
    
    Returns:
    - float: The maximum error estimated across all binning iterations.
    """
    if workingNdim <= 1:
        raise Exception('Not enough points MC steps were used for the binning method, please increase the number of MC steps')
    
    errors = [errorpropagation(data)]

    for i in range(1, workingNdim):
        # Efficient binning using reshape and mean along the new axis
        data = data[:2 * (len(data) // 2)].reshape(-1, 2).mean(axis=1)
        errors.append(errorpropagation(data))

    return max(errors)

def calculateError_byBinning(arr):
    """
    Calculate the mean and its maximum error estimate for a dataset using the binning method.
    
    Parameters:
    - arr: numpy.ndarray, the array of data points for which to calculate the mean and error estimate. Should be a 1D array.
    
    Returns:
    - tuple: A tuple containing the calculated mean and the maximum error estimate.
    """
    workingNdim = int(np.log2(len(arr)))
    trunc = len(arr) - 2**workingNdim
    arr = arr[trunc:]
    mean = np.mean(arr)
    standardError = maxError_byBinning(mean, arr, workingNdim - 6)
    return mean, standardError

def estimate_additional_steps():
    #TODO: implement function that estimate addtional steps required to reduce the error to 1% of the mean value 
    pass

class simulation_state():
    def __init__(self,result_dir="Result"):
        self.result_dir = result_dir
        self.path = {
            "params":os.path.join(result_dir, "params.json"),   
            "PE":os.path.join(result_dir, "pe.npy"),
            "POS":os.path.join(result_dir, "pos.npy"),
            "FORCES":os.path.join(result_dir, "forces.npy")
            }
        self.params = json.load(open(self.path["params"]))
        self.PE = np.load(self.path["PE"]) # shape (simulation_steps, P)
        self.POS = np.load(self.path["POS"]) # shape (simulation_steps, P, 4, 3)
        self.FORCES = np.load(self.path["FORCES"])  # shape (simulation_steps, P, 4, 3)
        self.simulation_steps = self.PE.shape[0]
        self.P = self.PE.shape[1]


    def analyze(self):
        self.compute_PE()
        self.compute_K_virial()
        self.compute_H2O_structure()
        self.compute_E_virial()
        self.compute_error_bar()
        self.save()
        self.print_log()

    def save(self):
        np.savez(os.path.join(self.result_dir, "result.npz"), 
                 geometry=self.geometry, 
                 E_virial_average= np.array(self.E_virial_avg), 
                 error_bar= np.array(self.error_bar))
        
    def print_log(self):
        print (f"E_virial: {self.E_virial_avg}")
        print(f"Error bar: {self.error_bar}, {self.error_percent} %")

    def compute_PE(self):
        self.pe = np.sum(self.PE, axis=1) # shape (simulation_steps,)

    def compute_K_virial_slow(self):
        self.K_virial = np.zeros(self.simulation_steps)
        for i in range(self.simulation_steps):
            K_virial = 0
            for beadi in range(self.P):
                posi = self.POS[i, beadi]
                forces = self.FORCES[i, beadi]
                for j in range(4):
                    K_virial -= np.dot(posi[j], forces[j])
            self.K_virial[i] = K_virial

    def compute_K_virial(self):
        # Adjust the einsum path to account for the num_atoms dimension
        # Now, the einsum string indicates:
        # - 'ijkl,ijkl->i' performs dot product across the last dimension (3 for x, y, z components),
        #   sums over all num_atoms and all beads for each step, resulting in one value per step
        # This assumes self.POS.shape and self.FORCES.shape are (steps, beads, num_atoms, 3)
        K_virial_all_steps = -np.einsum('ijkl,ijkl->i', self.POS, self.FORCES)
        self.K_virial = K_virial_all_steps

    def compute_H2O_structure_slow(self):
        for i in range(self.simulation_steps):
            for beadi in range(self.P):
                posi = self.POS[i, beadi]
                bead_rOH2=np.linalg.norm(posi[0]-posi[1])
                bead_rOH1=np.linalg.norm(posi[0]-posi[2])
                bead_rHH=np.linalg.norm(posi[2]-posi[1])
                bead_angle=np.arccos(np.dot(posi[0]-posi[2],posi[0]-posi[1])/bead_rOH1/bead_rOH2)*180./np.pi
                self.POS[i, beadi] = np.array([bead_rOH1*10., bead_rOH2*10., bead_rHH*10., bead_angle])

    def compute_H2O_structure(self):
        POS = self.POS 
        rOH2 = np.linalg.norm(POS[:, :, 0] - POS[:, :, 1], axis=-1)
        rOH1 = np.linalg.norm(POS[:, :, 0] - POS[:, :, 2], axis=-1)
        rHH = np.linalg.norm(POS[:, :, 2] - POS[:, :, 1], axis=-1)
        vec_OH1 = POS[:, :, 0] - POS[:, :, 2]
        vec_OH2 = POS[:, :, 0] - POS[:, :, 1]
        cos_angle = np.einsum('ijk,ijk->ij', vec_OH1, vec_OH2) / (rOH1 * rOH2)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180. / np.pi
        self.geometry = np.stack((rOH1 * 10., rOH2 * 10., rHH * 10., angle), axis=-1) #shape (simulation_steps, P, 4)
    
    def compute_E_virial(self):
        self.E_virial = (.5*self.K_virial/self.P+self.pe/self.P)/4.184
        self.E_virial_avg = np.mean(self.E_virial)

    def compute_error_bar(self):
        _, error_bar = calculateError_byBinning(self.E_virial)
        check_error_percent = error_bar/self.E_virial_avg*100
        self.error_bar = error_bar
        self.error_percent = check_error_percent





    
    
