# test

import numpy as np
import numpy.linalg as lg

########################################
###            Functions             ###
########################################
def H_k(k_vec, p=1, q=2):
    kx=k_vec[0]
    ky=k_vec[1]
    # Initialize the matrix
    M = np.zeros((q, q), dtype=complex)

    # Fill in the matrix
    for i in range(q):
        # Main diagonal
        M[i, i] =  2 * np.cos(ky +2*np.pi*(p/q) * (i+1))
    # Upper diagonal
        if i + 1 < q:
            M[i, i + 1] = np.exp(kx*1j)
    # Lower diagonal
        if i - 1 >= 0:
            M[i, i - 1] = np.exp(-kx*1j)

    M[0,q-1]= np.exp(-1j*kx)
    M[q-1,0]= np.exp(1j*kx)  

    if q==2:
        M[0,1]=np.exp(+kx*1j)+np.exp(-kx*1j) 
        M[1,0]=np.exp(+kx*1j)+np.exp(-kx*1j)  
    M=M+M.conj().T      
    return M

def build_U(vec1,vec2):
    """function to calculate the iner product of two
    eigenvectors divided by the norm:
    
    U = <psi|psi+mu>/|<psi|psi+mu>|

    input:
    ------
    vec, vec2: vectors complex.

    return:
    -------
    U: scalar complex number

    """

    # U = <psi|psi+mu>/|<psi|psi+mu>|
    in_product = np.dot(vec1,vec2.conj())

    U = in_product / np.abs(in_product)

    return U
############################################################

def latF(k_vec, Dk, dim):
    """calulate lattice field using the definition: F12 = ln[
    U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ] for each
    k=(kx,ky) point, four U must be calculated.  The lattice
    field has the same dimension of the number of energy
    bands.
    
    input:
    ------
    k_vec:vec(2), float, (kx,ky).
    Dk: vec(2), float, (Dkx,Dky),
    dim:integer,  dim of H(k)
    
    return:
    -------
    F12:vec(dim), complex, lattice field corresponding to each band.
    E_sort: vec(dim) float, eigenenergies.
    """

    # Here we calculate the band structure and sort
    # them from low to high eigenenergies

    k = k_vec
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    E_sort = E[idx]
    psi = aux[:,idx]

    k = np.array([k_vec[0]+Dk[0], k_vec[1]], float)
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    psiDx = aux[:,idx]

    k = np.array([k_vec[0], k_vec[1]+Dk[1]], float)
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    psiDy = aux[:,idx]

    k = np.array([k_vec[0]+Dk[0], k_vec[1]+Dk[1]], float)
    E, aux = lg.eig( H_k(k) )
    idx = E.argsort()
    psiDxDy = aux[:,idx]

    U1x = np.zeros((dim), dtype=complex)
    U2y = np.zeros((dim), dtype=complex)
    U1y = np.zeros((dim), dtype=complex)
    U2x = np.zeros((dim), dtype=complex)

    for i in range(dim):
        U1x[i] = build_U(psi[:,i], psiDx[:,i] )
        U2y[i] = build_U(psi[:,i], psiDy[:,i] )
        U1y[i] = build_U(psiDy[:,i], psiDxDy[:,i] )
        U2x[i] = build_U(psiDx[:,i], psiDxDy[:,i] )

    F12 = np.zeros((dim), dtype=complex)

    F12 = np.log( U1x * U2x * 1./U1y * 1./U2y)

    return F12, E_sort

##################################################
###             Main program                   ###
##################################################

x_res = 50
y_res = 50
q = 2
Nd = q                          # dimension of the Hamiltonian

Dx = (2.*np.pi/3.)/x_res
Dy = (2.*np.pi)/y_res
Dk = np.array([Dx,Dy], float)

LF = np.zeros((Nd), dtype=complex)
LF_arr = np.zeros((Nd,x_res, y_res), dtype=float) # plotting array
sumN = np.zeros((Nd), dtype=complex)
E_k = np.zeros((Nd), dtype=complex)
chernN = np.zeros((Nd), dtype=complex)

for ix in range(x_res):

    kx = ix*Dx
    for iy in range(y_res):

        ky = iy*Dy

        k_vec = np.array([kx,ky], float)
        LF, E_k = latF(k_vec, Dk, Nd)

        sumN += LF

        # save data for plotting
        LF_arr[:,ix,iy] = -LF.imag/(2.*np.pi) 

chernN = sumN.imag/(2.*np.pi)
print("Chern number associated with each band: ", chernN)
