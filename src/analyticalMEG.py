#import everything
import time
import csv
import numpy as np
import math
from numpy import linalg as LA
import scipy.io

def analyticalMEGsolution(coil_pos, dipoles, n) :
    start_time = time.time()
    #fix constants
    c =[127, 127, 127] #center of the sphere
    mu = 4 * math.pi * 1e-10
    #import coil positions
    coils =[]
    with open(coil_pos, newline ='') as inputfileC :
        for row in csv.reader(inputfileC, dialect = 'excel-tab') :
            coils.append(list(np.array([float(x) for x in row]) - np.array(c)))
    if n == 1 :
        #import dipole positions
        dipolesPOS =[]
        dipolesMOM =[]
        with open(dipoles, newline ='') as inputfileD :
            for row in csv.reader(inputfileD) :
                dipolesPOS.append(list(np.array([float(x) for x in row[0].split()[0 : 3]]) - np.array(c)))
                dipolesMOM.append(list(np.array([float(x) for x in row[0].split()[3 : 6]])))
    else :
        dipolesPOS =[dipoles[i] for i in(0, 1, 2)]
        dipolesMOM =[dipoles[i] for i in(3, 4, 5)]
    #compute analytical Bp, B(and Bs) : primary, full(and secondary) magnetic field(Sarvas' formula)
    B =[]
    Bp =[]
    for k, (ro, q) in enumerate(zip(dipolesPOS, dipolesMOM)) :
        Bhelp =[]
        Bphelp =[]
        for r in coils :
            a = np.array(r) - np.array(ro)
            aa = a / math.pow(LA.norm(a), 3)
            F = LA.norm(a) *(LA.norm(a) * LA.norm(r) + LA.norm(r) * LA.norm(r) - np.dot(np.array(ro), np.array(r)))
            gradF =((LA.norm(a) * LA.norm(a)) / LA.norm(r) + np.dot(a, np.array(r)) / LA.norm(a) + 2. *(LA.norm(a) + LA.norm(r))) * np.array(r) -(LA.norm(a) + 2. * LA.norm(r) +(np.dot(a, np.array(r)) / LA.norm(a))) * np.array(ro)
            bla =(mu /(4. * math.pi * F * F)) *(F * np.cross(q, ro) -(np.dot(np.cross(q, ro), np.array(r))) * gradF)
            Bhelp.append(list(bla))
            Bphelp.append(list((mu /(4. * math.pi)) * np.cross(q, aa)))

        B.append(np.array([item for sublist in Bhelp for item in sublist]))
        Bp.append(np.array([item for sublist in Bphelp for item in sublist]))
    #extract the secondary component
    Bs = list(np.array(B) - np.array(Bp))
    #printing the time
    print("--- %s seconds ---" %(time.time() - start_time))
    #timing with 20K dipoles : -- - 611.4201629161835 seconds-- -
    #transform Bp, Bs, B in unique lists
    #Bp_py = [item for sublist in Bp for item in sublist]
    #Bs_py = [item for sublist in Bs for item in sublist]
    #B_py = [item for sublist in B for item in sublist]
    return Bp, Bs, B #Bp_py, Bs_py, B_py
