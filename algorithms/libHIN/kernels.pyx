## efficient implementation of some of the graphlet kernels.

import numpy as np
cimport numpy as np

def shortest_path_kernel(np.ndarray A):
## init types
    
    cdef int n = A.shape[0]
    n = A.shape[0]
    cdef int x, i, j
    cdef np.ndarray Sp = np.zeros([n, n], dtype=np.int)
    
    # # init
    Sp = np.zeros(shape = (n,n))
    
    for i in range(n):
        for j in range(n):
            if A[i,j] == 0 and i!=j:
                Sp[i,j]=float("inf")                
            else:
                Sp[i,j]=A[i,j]

    # shortest paths
    # for k in range(n):
    #     for i in range(n):
    #         for j in range(n):
    #      if Sp[i,j] > Sp[i,k] + Sp[k,j]:
    #          #Sp[i,j]=Sp[i,k]+Sp[k,j]
    # return Sp

# if __name__ == "__main__":
#     mx = np.random.randint(2, size=(5, 5))
#     shortest_path_kernel(mx)
