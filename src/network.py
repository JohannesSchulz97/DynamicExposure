import numpy as np
import pandas as pd
import sys 

class Network(object):

    '''
        relevant parameters: 
        - dirichlet             # the lower the more peaked the distribution. 
                                This implies fewer mixed community membership vectors, i.e. problem is harder for higher values
        - seed                  # the seed of the random number generator for reproducibility
        - probability_ratio     # ratio between probability of in-community and out-community links
        - overlapping           # ratio of nodes with mixed community memberships
        - correlation           # correlation between membership vector u and membership vector v
        - N
        - K
        - structure             # at the moment assortative or disassortative

        relevant for the actual data generation
        
        - T
        - p_T                   # probability that exposure has happened between two nodes for the last network instance
        - avg_degree            # for the data tensor before applying exposure


        Configurations for the actual experiments: 
        - p_T=1 or 0.999 if 1 does not work mathematically 
        - avg_degree = 10
        - T = [30, 100, 300]
        - N = 100
        - K = 3
        - dirichlet = 1
        - prob_ratio = 0.1
        - seed = [10 different random integer seeds]
        - correlation = 1
        - overlapping = 1
        - structure = "assortative"


    '''

    RANDOM_SEEDS= [697752728, 4190089612, 1176914559, 3077924848, 315917623, 2544020234,
                    1077758578, 4071300106, 534591752, 3553386411]

    def __init__(self, N=100, K=3, seed=42, dirichlet=1, prob_ratio=0.1, overlapping=1, correlation=1):
        self.random_state = np.random.RandomState(seed) 
        self.probability_ratio = prob_ratio
        self.overlapping = overlapping
        self.correlation = correlation
        self.N = N
        self.K = K
        self.structure = "assortative"
        self.dirichlet = dirichlet
        self.initialize_latent_variables()


    def initialize_latent_variables(self):
        self.membership_vectors()
        self.affinity_matrix()
        self.compute_lambda()


    def get_u(self): 
        return self.u

    def get_v(self): 
        return self.v

    def get_w(self): 
        return self.w
    
    def get_mu(self): 
        return self.mu
    
    def get_Z(self):
        return self.Z

    def membership_vectors(self):

        size_mixed = int(np.round(self.overlapping * self.N))
        ind_mixed= np.random.choice(np.arange(self.N), size_mixed, replace=False)
        ind_hard = np.delete(np.arange(self.N), ind_mixed)


        u = np.zeros(shape=(self.N,self.K))
        v = np.zeros_like(u)
        
        # sample hard memberships uniformly at random
        for ind in ind_hard:
            u[ind, np.random.randint(self.K)] = 1
            if np.random.random() < self.correlation: 
                v[ind] = u[ind]
            else:
                v[ind, np.random.randint(self.K)] = 1
        
        u[ind_mixed] = self.random_state.dirichlet(self.dirichlet*np.ones(self.K), size=len(ind_mixed))
        v[ind_mixed] = self.correlation * u[ind_mixed] + (1-self.correlation) * self.random_state.dirichlet(self.dirichlet*np.ones(self.K), size=len(ind_mixed))

        # sort u and v based on hard memberships in u 
        hard_memberships_u = np.argmax(u, axis=1)
        # sort u and v based on hard memberships in u 
        hard_memberships_u = np.argmax(u, axis=1)
        self.u = np.array([x for x,_ in sorted(zip(u, hard_memberships_u), key=lambda pair: pair[1])])
        self.v = np.array([x for x,_ in sorted(zip(v, hard_memberships_u), key=lambda pair: pair[1])])
        

    
    """
    
    The affinity matrix can have different structures: 
    
    - assortative: nodes of the same community have a higher probability to be connected
    - disassortative: nodes of the same community have a lower probability to be connected
    - core-periphery: 
    - directed-biased: 
    
    """
    def affinity_matrix(self):
        
        K = self.K

        # ratio between low and high probability
        prob_ratio = self.probability_ratio
        high_prob = 1
        low_prob = high_prob * prob_ratio
        
        w = low_prob * np.ones((K,K))  
        np.fill_diagonal(w, high_prob * np.ones(K))  

        self.w = w


    "scale affinity matrix w to get the correct expected average degree"
    def adjust_for_avg_degree(self, avg_degree):
        N = self.N
        temp = self.lam.copy()
        np.fill_diagonal(temp, 0)
        scale = avg_degree * N / temp.sum()
        self.w = scale * self.w
        self.lam = scale * self.lam

    def compute_lambda(self):
        self.lam = self.u @ self.w @ self.v.T 

    def generate_A0(self, T, avg_degree, verbose=False):
        self.adjust_for_avg_degree(avg_degree=avg_degree)
        data = self.random_state.poisson([self.lam]*T)
        for t in range(T): 
            np.fill_diagonal(data[t], 0)
        if verbose: 
            print(f"expected average degree for A0: {avg_degree}")
            print(f"actual average degree for A0: {data.sum() / self.N / T}")
        return data


    def apply_exposure(self, data, K_exp): 
        T, N = data.shape[0], data.shape[1]
        mu = self.random_state.dirichlet(self.dirichlet * np.ones(K_exp), size=N)
        self.mu = mu
        mu_ij = np.einsum('ik,jk->ij', mu, mu)
        Z = self.random_state.binomial(n=1, size=(T,N,N), p=mu_ij)
        self.Z = Z
        return data*Z

