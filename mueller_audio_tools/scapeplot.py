import librosa

import numpy as np
import librosa 
import math

'''
Source --
  * https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_AudioThumbnailing.html

Authored by: 
  Meinard Mueller, Angel Villar-Corrales
Arranged by:
  Wen-Yi Hsiao, Shih-Lun Wu
'''

# ------------------------------------------------------------ #
# Fitness Scape Plot Computation
# ------------------------------------------------------------ #
def normalization_properties_SSM(S):
    """Normalizes self-similartiy matrix to fulfill S(n,n)=1
    Yields a warning if max(S)<=1 is not fulfilled
   
    Notebook: C4/C4S3_AudioThumbnailing.ipynb 
    """    
    N = S.shape[0]
    for n in range(N): 
        S[n,n] = 1
        max_S = np.max(S)
    if max_S>1:
        print('Normalization condition for SSM not fulfill (max > 1)')
    return S


def compute_accumulated_score_matrix(S_seg):
    """Compute the accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S_seg: submatrix of an enhanced and normalized SSM S 
                 Note: S must satisfy S(n,m) <= 1 and S(n,n) = 1
        
    Returns:
        D: Accumulated score matrix 
        score: Score of optimal path family 
    """
    inf = math.inf  
    N =  S_seg.shape[0]
    M =  S_seg.shape[1]+1
    
    # Iinitializing score matrix
    D = -inf*np.ones((N,M), dtype=np.float64)
    D[0,0] = 0.
    D[0,1] = D[0,0]+S_seg[0,0]

    # Dynamic programming
    for n in range(1, N):
        D[n,0] = max( D[n-1,0], D[n-1,-1] )    
        D[n,1] = D[n,0] + S_seg[n, 0]
        for m in range(2, M):
            D[n, m] = S_seg[n, m-1] + max( D[n-1, m-1], D[n-1, m-2], D[n-2, m-1] )
            
    # Score of optimal path family
    score = np.maximum( D[N-1,0], D[N-1,M-1] )
    
    return D, score

def compute_optimal_path_family(D):
    """Compute an optimal path family given an accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        D: Accumulated score matrix

    Returns
        P: Optimal path family consisting of list of paths 
           (each path being a list of index pairs)
    """
    # Initialization
    inf = math.inf
    N = int(D.shape[0])
    M = int(D.shape[1])
    
    path_family = []
    path = []
    
    n = N - 1
    if( D[n,M-1]<D[n,0] ):
        m = 0
    else:
        m = M-1
        path_point = (N-1, M-2)
        path.append(path_point)
    
    # Backtracking 
    while n > 0 or m > 0:

        # obtaining the set of possible predecesors given our current position
        if(n<=2 and m<=2):
            predecessors = [(n-1,m-1)]
        elif(n<=2 and m>2):
            predecessors = [(n-1,m-1),(n-1,m-2)]
        elif(n>2 and m<=2):
            predecessors = [(n-1,m-1),(n-2,m-1)]
        else:
            predecessors = [(n-1,m-1),(n-2,m-1),(n-1,m-2)]
        
        # case for the first row. Only horizontal movements allowed
        if n == 0:
            cell = (0, m-1)
        # case for the elevator column: we can keep going down the column or jumping to the end of the next row
        elif m == 0:
            if( D[n-1, M-1] > D[n-1, 0] ):
                cell = (n-1, M-1)
                path_point = (n-1, M-2)
                if(len(path)>0):
                    path.reverse()
                    path_family.append(path)
                path = [path_point]
            else:
                cell = (n-1, 0)
        # case for m=1, only horizontal steps to the elevator column are allowed
        elif m == 1:
            cell=(n,0)          
        # regular case
        else:
        
            #obtaining the best of the possible predecesors
            max_val = -inf
            for i in range(len(predecessors)):
                if( max_val<D[predecessors[i][0],predecessors[i][1]] ):
                    max_val = D[predecessors[i][0],predecessors[i][1]]
                    cell = predecessors[i]  
                    
            #saving the point in the current path
            path_point = (cell[0],cell[1]-1)
            path.append(path_point)        
            
        (n, m) = cell
    
    # adding last path to the path family
    path.reverse()
    path_family.append(path)
    path_family.reverse()
    
    return path_family

def compute_induced_segment_family_coverage(path_family):
    """Compute induced segment family and coverage from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family: Path family

    Returns
        segment_family: Induced segment family
        coverage: Coverage of path family
    """
    num_path = len(path_family)
    coverage = 0
    if num_path>0:
        segment_family = np.zeros((num_path, 2), dtype=int)
        for n in range(num_path):
            segment_family[n,0] = path_family[n][0][0]
            segment_family[n,1] = path_family[n][-1][0]
            coverage = coverage + segment_family[n,1] - segment_family[n,0] + 1
    else:
        segment_family = np.empty
        
    return segment_family, coverage

def compute_fitness(path_family, score, N):
    """Compute fitness measure and other metrics from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family: Path family 
        score: Score of path family 
        N: Length of feature sequence

    Returns
        fitness: Fitness
        score: Score
        score_n: Normalized score
        coverage: Coverage
        coverage_n: Normalized coverage
        path_family_length: Length of path family (total number of cells)
    """
    eps = 1e-16
    num_path = len(path_family)
    M = path_family[0][-1][1] + 1
    
    # Normalized score
    path_family_length = 0 
    for n in range(num_path):
        path_family_length = path_family_length + len(path_family[n])
    score_n = (score - M) / (path_family_length + eps)

    # Normalized coverage
    segment_family, coverage = compute_induced_segment_family_coverage(path_family)
    coverage_n = (coverage - M) / (N + eps)
    
    # Fitness measure
    fitness = 2 * score_n * coverage_n / (score_n + coverage_n + eps)
    
    return fitness, score, score_n, coverage, coverage_n, path_family_length


def compute_fitness_scape_plot(S):
    """Compute scape plot for fitness and other measures

    Notebook: /C4/C4S3_ScapePlot.ipynb

    Args:
        S: Self-similarity matrix 

    Returns:
        SP_all: Vector containing five different scape plots for five measures
            (fitness, score, normalized score, coverage, normlized coverage)
            (encoded as start-duration matrix)
    """
    N = S.shape[0]
    SP_fitness = np.zeros((N,N))    
    SP_score = np.zeros((N,N))
    SP_score_n = np.zeros((N,N))
    SP_coverage = np.zeros((N,N))
    SP_coverage_n = np.zeros((N,N))
    
    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            S_seg = S[:,start:start+length_minus_one+1]         
            D, score = compute_accumulated_score_matrix(S_seg)
            path_family = compute_optimal_path_family(D)
            fitness, score, score_n, coverage, coverage_n, path_family_length = \
                compute_fitness(path_family, score, N)
            SP_fitness[length_minus_one,start]= fitness
            SP_score[length_minus_one,start]= score
            SP_score_n[length_minus_one,start]= score_n
            SP_coverage[length_minus_one,start]= coverage
            SP_coverage_n[length_minus_one,start]= coverage_n
    SP_all = [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
    return SP_all
