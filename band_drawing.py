import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append(r'C:\Users\EFE\OneDrive\Masaüstü\PHYS400_Chern\chern-master')  # Adjust the path as necessary
import chern



# Define function of Harper matrix 
def H(p, q, kx, ky):
    
    # Initialize the matrix
    M = np.zeros((q, q), dtype=complex)

    # Fill in the matrix
    for i in range(q):
        # Main diagonal
        M[i, i] =  2 * np.cos(ky +2*np.pi*(p/q) * i)
    # Upper diagonal
        if i + 1 < q:
            M[i, i + 1] = np.exp(+kx*i*1j)
    # Lower diagonal
        if i - 1 >= 0:
            M[i, i - 1] = np.exp(-kx*i*1j)

    M[0,q-1]= np.exp(-1j*ky)
    M[q-1,0]= np.exp(1j*ky)  

    if q==2:
        M[0,1]=np.exp(+kx*i*1j)+np.exp(-kx*i*1j) 
        M[1,0]=np.exp(+kx*i*1j)+np.exp(-kx*i*1j)    
    return M

Q=2
p=1



ev=np.linalg.eig(H(p,2,4*np.pi/3,4*np.pi/3))[1]

for q in range(1,Q+1):
    xline=np.linspace(-4*np.pi/3,4*np.pi/3)
    yline=np.linspace(-4*np.pi/3,4*np.pi/3)
    X, Y = np.meshgrid(xline, yline)
    Z = np.zeros(X.shape + (q,))

    # Populate Z with eigenvalues for each (kx, ky) pair
    for i in range(len(xline)):
        for j in range(len(yline)):
            eigenvals = np.linalg.eigvalsh(H(p, q, X[i, j], Y[i, j]))
            for k in range(q):
                Z[i, j, k] = eigenvals[k]

    #x1 = np.linalg.eigvalsh(H(1,3,xline, yline))
    #x2= np.linalg.eigvalsh(H(1,3,kx=2*np.pi/3, ky=2*np.pi/3))
    #Z=np.transpose(Z)

    # Combined surface plot for all eigenvalue levels, with custom colors
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['coolwarm', 'viridis', 'plasma','inferno', 'magma', 'cividis']  # Custom colors for each eigenvalue level

    for k in range(np.shape(Z)[-1]):
        surf = ax.plot_surface(X, Y, Z[:, :, k], cmap=colors[k], edgecolor='none', linewidth=0, antialiased=True, alpha=0.7)

    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Eigenvalue')
    ax.set_title('Combined Eigenvalues of Harper\'s Equation')

    # Adding a legend is tricky for 3D surface plots, so we'll use a custom approach
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Eigenvalue Level {k+1}',
                            markerfacecolor=cm.get_cmap(colors[k])(0.5), markersize=10) for k in range(np.shape(Z)[-1])]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()
    
