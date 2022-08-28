import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
import os

import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

from scipy import spatial
import scipy
from sklearn.preprocessing import normalize
from sklearn import metrics



COLORS = {0:'b',1:'r',2:'g',3:'orange',4:'black',5:'magenta', 6:'cyan', 7: 'yellow', 8: 'grey'}



def build_edgelist(A, t):
    """
        Build the edgelist for a given layer A_t in DataFrame format.
        INPUT
        ----------
        A : list
            Adjacency matrix
        l : int
            Layer number.
        OUTPUT
        -------
        df_res : DataFrame
                Pandas DataFrame with edge information about a given layer.
    """
    A = csr_matrix(A)
    A_coo = A.tocoo()
    data_dict = {'source': A_coo.row, 'target': A_coo.col, 'T'+str(t): A_coo.data}
    dft = pd.DataFrame(data_dict, dtype="Int64")
    
    return dft


def write_data(folder, file, data):
    """
        Save the adjacency tensor to a file.
        Default format is space-separated .csv with T+2 columns: source_node target_node edge_l0 ... edge_lT
        INPUT
        ----------
        A : list
            T*N*N numpy multitensor representing the multi-tensor 
        name : str
                filename to store the adjacency tensor.
        out_folder : str
                    Path to store the adjacency tensor. 
    """

    outfile = file + '.csv'
    if not os.path.exists(folder):
        os.makedirs(folder)

    
    T = data.shape[0]
    for t in range(T):
        dft = build_edgelist(data[t], t)
        if t==0: 
            df=dft
        else: 
            df = df.merge(dft, on=["source", "target"], how="outer")

    
    df.to_csv(folder + outfile, index=False, sep=' ')
    print(f'Adjacency matrix saved in: {folder + outfile}')
    
def read_data(folder, file): 
    """
    Recover the data as a numpy array from a file path
    INPUT
    ----------
    file_path : location of csv file that represents our data 
    """
    file_path = folder + file + ".csv"
    df = pd.read_csv(file_path, sep=' ', dtype="Int64")
    indices = df[['source', 'target']].to_numpy()
    T = len(df.columns) - 2
    N = np.amax(indices) + 1
    df = df.set_index(["source", "target"])
    df = df.fillna(0)
    
    data = np.zeros((T, N, N))
    for i, row in df.iterrows():
        for t in range(T): 
            data[t,i[0],i[1]] = row["T"+str(t)]
    return data

def write_params(folder, file, u, v, w, mu=None, Q=None, exp=True): 
    if exp: 
        np.savez(folder+"params_" + file, mu=mu, Q=Q, u=u, v=v, w=w)
    else: 
        np.savez(folder+"params_" + file, u=u, v=v, w=w)

def write_training(folder, file, losses, theta_errors, exp_errors, exp=True):
    if exp: 
        np.savez(folder+"training_" + file, losses=losses, theta_errors=theta_errors, exp_errors=exp_errors)
    else: 
        np.savez(folder+"training_" + file, losses=losses, theta_errors=theta_errors)

def read_params(folder, file, exp=True):
    if exp: 
        params = np.load(folder + "params_exp_" + file + ".npz")
    else: 
        params = np.load(folder + "params_noexp_" + file + ".npz")
    return params

def read_training(folder, file, exp=True):
    if exp: 
        losses = np.load(folder + "training_exp_" + file + ".npz")
    else: 
        losses = np.load(folder + "training_noexp_" + file + ".npz")
    return losses

def calculate_permutation_matrix(u, u_est): 
    N, K = u.shape 
    M = u_est.T @ u
    
    P = np.zeros((K,K))
    for k in range(K): 
        max_index = np.unravel_index(np.argmax(M), M.shape)
        P[max_index] = 1
        M[max_index[0], :] = -1
        M[:, max_index[1]] = -1
    return P


def permute_memberships(u, u_est, v, v_est, w_est=None):
    u_est = normalize(u_est)
    v_est = normalize(v_est)
    P_u = calculate_permutation_matrix(u, u_est)
    P_v = calculate_permutation_matrix(v, v_est)
    u_perm = u_est@P_u
    v_perm = v_est@P_v
    if w_est is None: 
        return u_perm, v_perm
    else: 
        w_perm = (P_u.T @ w_est) @ P_v
        return u_perm, v_perm, w_perm


"""
    calculate AUC score for estimation

    used for A and Z: 
    - A: compare A at certain indices (mask) with prediction i.e. lambda(mask)
    - Z: compare Z with prediction i.e. Q
"""
def calculate_AUC(data, pred, verbose=False, mask=None):
    if len(data.shape) == 3: 
        T, N = data.shape[0], data.shape[1]
        default_mask = np.ones((T,N,N))
        for t in range(T): 
            np.fill_diagonal(default_mask[t], 0)
    else: 
        N = data.shape[0]
        default_mask = 1 - np.identity(N)

    if mask is None: 
        mask = default_mask
    else: 
        mask = mask * default_mask
    
    # check if ground truth is all one
    '''
    fpr, tpr, _ = metrics.roc_curve(data[mask>0].flatten(), pred[mask>0].flatten())

    auc_score = metrics.auc(fpr, tpr)    
    '''
    auc_score = metrics.roc_auc_score(data[mask>0].flatten(), pred[mask>0].flatten())
    
    if verbose: 
        print(f"AUC score: {auc_score}\n")
        #print("TPR: ", tpr.mean())
        #print("FPR: ", fpr.mean())
    return auc_score

'''
    Mean absolute error metric for inference performance on Z
'''
def MAE(Z, Q):
    return np.abs(Z-Q).sum() / Z.size

'''
    evaluate inference performance on Z by calculating: 
    - the expectation of Q and based on that the average deviation of this expectation and the true exposure Z
    - the average variance of Q to get a measure of certainty on the expectation
'''
def evaluate_Q(Z, Q, verbose=False):
    T, N = Q.shape[0]-1, Q.shape[1]
    expectation_Q = (Q * np.array([[np.arange(0,T+1)]*N]*N).T).sum(axis=0)
    variance_Q = (Q * (np.array([[np.arange(0,T+1)]*N]*N).T - expectation_Q)**2).sum(axis=0)
    '''
        to ensure that exposure between nodes with themselves do not impact the result
        UPDATE: no longer necessary, because the diagonal elements in Q are modified to match Z
                in the InferenceMulti.fit() function
    
    np.fill_diagonal(expectation_Q, 0)
    np.fill_diagonal(variance_Q, 0)
    '''
    
    '''
        test if the computation of the expectation and the variance is correct
    '''
    '''
    for i in range(N): 
        for j in range(N): 
            if i==j: 
                continue
            exp_ij = (Q[:,i,j] * np.arange(0,T+1)).sum()
            var_ij = (Q[:,i,j] * [(t-expectation_Q[i,j])**2 for t in range(T+1)]).sum()
            assert (np.round(exp_ij, 5) == np.round(expectation_Q[i,j], 5)), f"error in expectation for i={i}, j={j}: {exp_ij}, {expectation_Q[i,j]}"
            assert (np.round(var_ij, 5) == np.round(variance_Q[i,j], 5)), f"error in variance for i={i}, j={j}: {var_ij}, {variance_Q[i,j]}"
    '''
    if verbose: 
        print(f"Z: \n {Z}")
        print(f"expectation Q: \n {np.round(expectation_Q, 2)}")
        print(f"variance Q: \n {np.round(variance_Q, 2)}")
    mean_deviation = np.abs(Z-expectation_Q).sum() / (N**2 -N)
    mean_variance = variance_Q.sum()/ (N**2 - N)
    return np.round(mean_deviation, 3), np.round(mean_variance, 3)



def switch_representation(Z, T=None): 
    if len(Z.shape) == 2: 
        N = Z.shape[0]
        if T is None: 
            raise Exception("if Z is 2 dimensional, T has to be given")
        temp = np.zeros((T+1, N, N))
        exp_matrix = np.minimum(Z, T)
        for i in range(N): 
            for j in range(N):
                temp[exp_matrix[i,j],i,j] = 1
        return temp
    if len(Z.shape) == 3: 
        return np.argmax(Z>0, axis=0)   


def extract_pie_chart_properties(memberships, threshold=0):
    groups = np.where(memberships > threshold)[0]
    wedge_sizes = memberships[groups]
    wedge_colors = [COLORS[group] for group in groups]
    return wedge_sizes, wedge_colors

def extract_mask(T, N, Nfolds=5, seed=24, symmetric=True): 
    masks = np.zeros((Nfolds,T,N,N), dtype=bool)
    random_state = np.random.RandomState(seed)
    folds = random_state.choice(np.arange(0,Nfolds), size=(T,N,N))
    for t in range(T): 
        if symmetric: 
            folds[t] = np.tril(folds[t],-1) + np.tril(folds[t],-1).T
        np.fill_diagonal(folds[t], Nfolds)
        for l in range(Nfolds): 
            masks[l,t] = (folds[t]==l)
    return masks


    
"""
visualize graph instance
lv contains infered membership vector. If lv!=None two graph are plotted next to each other.
"""


def plot_graph(A, memberships, directed=True, title="Memberships", save_as=None):
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph) 
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    node_size = [np.log(graph.in_degree[i] + graph.out_degree[i] + 1)*50 for i in list(graph.nodes())]
    position = nx.spring_layout(graph, iterations=1000)
    #position = nx.fruchterman_reingold_layout(graph)

    nx.draw_networkx_edges(graph, position, width=0.2, arrows=directed, ax=ax)

    for i,n in enumerate(graph.nodes):
        wedge_sizes, wedge_colors = extract_pie_chart_properties(memberships[i])
        ax.pie(wedge_sizes, normalize=True, center=position[n], colors = wedge_colors, radius=node_size[i]*0.0001) 
        ax.axis("equal")
        ax.set_title(title)

    plt.axis('off')
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


def plot_adjacency_matrix(A, lam):
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    axs[0].imshow(A)
    axs[0].set_title("Adjacency Matrix", fontsize=15)
    # if the correlation between u and v is low, structure can look bad (both are sorted based on hard memberships in u)
    axs[1].imshow(lam)
    axs[1].set_title("Expected Adjacency Matrix", fontsize=15)
    plt.show()


def plot_losses(probabilities, title="Training"):
    fig, ax = plt.subplots(1,1, figsize=(10, 4))
    plt.title(title)
    ax.plot(probabilities)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.grid()
    plt.tight_layout()
    plt.show()

def accuracy_over_time(Z,Q): 
    T = Z.shape[0]
    accuracies = []
    for t in range(T): 
        acc_t = 1- MAE(Z[t], Q[t])
        accuracies.append(acc_t)
    return accuracies

def mae_over_time(Z, Q): 
    T = Z.shape[0]
    mae_scores = []
    for t in range(T): 
        mae_t = MAE(Z[t], Q[t])
        mae_scores.append(mae_t)
    fig, ax = plt.subplots(1,1, figsize=(10, 4))
    plt.title("MAE scores")
    ax.plot(mae_scores)
    ax.set_xlabel("t")
    ax.set_ylabel("Mean Absolute Error")
    ax.grid()
    plt.tight_layout()
    plt.show()



"""
    Permuting the overlap matrix so that the groups from the two partitions correspond
    U0 has dimension NxK, reference membership.
"""

def evaluate_memberships(u, u_est, v, v_est, verbose=False): 
    u_est, v_est = permute_memberships(u, u_est, v, v_est)
    N = u.shape[0]
    acc_u = 0
    acc_v = 0
    count_u = 0
    count_v = 0
    for i in range(N): 
        if u_est[i].sum() != 0: 
            count_u += 1
            acc_u += 1-spatial.distance.cosine(u[i], u_est[i])
        if v_est[i].sum() != 0:
            count_v += 1
            acc_v += 1-spatial.distance.cosine(v[i], v_est[i])
    if count_u != 0:
        acc_u /= N#count_u
    if count_v != 0:
        acc_v /= N#count_v
    if verbose: 
        print(f"Acccuracy of predictions: \n\t u: {acc_u}\n\t v: {acc_v}")
        print(f"Zero estimate counts: \n\t u: {N-count_u}\n\t v: {N-count_v}\n\n\n")
    return acc_u, acc_v, count_u, count_v

def evaluate_mu(mu, mu_est, verbose=False, eps=1e-20): 
    N = mu.shape[0]
    acc_mu = 0
    for i in range(N): 
        acc_mu += 1-spatial.distance.cosine(mu[i] + eps, mu_est[i] + eps)
    acc_mu /= N
    if verbose: 
        print(f"Acccuracy mu: {acc_mu}\n")
    return acc_mu



import collections
'''
    created nested dictionary with arbitrary depth
'''
def makehash():
    return collections.defaultdict(makehash)

    

def average_exposure(b, a = 2., T = 10, M = 10, prng = None, seed = 10, N = 100):
    
    if prng is None: prng = np.random.RandomState(seed)
        
    mu = prng.beta(a,b,(M,N)) 
    muij = np.einsum('ai,aj-> aij', mu,mu)
    muijT = np.power(1 - muij,T)
    
    for k in range(M): 
        np.fill_diagonal(muijT[k], 0)
    p_T = 1 - np.sum(muijT) / (M * N * (N-1))
    return p_T


def determine_b_beta(T, p_T, lower_bound=1e-5, upper_bound=1000, tol=1e-3, verbose=False): 
    middle = np.round((lower_bound + upper_bound) / 2, 4)
    p = average_exposure(b=middle, T=T, M=1000)
    error = np.abs(p_T - p)
    if error < tol: 
        if verbose: 
            print(f"binary search converged for T={T}, p_T={p_T} and b_beta={middle} with an approximation error of {np.round(error, 5)} and average exposure of {np.round(p, 4)}")
        return middle
    if p < p_T: 
        return determine_b_beta(T, p_T, lower_bound=lower_bound, upper_bound=middle, tol=tol, verbose=verbose)
    if p > p_T: 
        return determine_b_beta(T, p_T, lower_bound=middle, upper_bound=upper_bound, tol=tol, verbose=verbose)


def sample_timespace_logarithmically(start, end, num=20):
    return np.unique(np.round(np.geomspace(start, end, num=num))).astype(int)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def lighten_color(color, amount=0.3):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    #return colorsys.hls_to_rgb(c[0], c[1], amount*c[2])
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def print_bold(text): 
    print('\033[1m' + text + '\033[0m')