import numpy as np
import pandas as pd
import sys 

class Network(object):


    RANDOM_SEEDS= [697752728, 4190089612, 1176914559, 3077924848, 315917623, 2544020234,
                    1077758578, 4071300106, 534591752, 3553386411]

    def __init__(self, N=100, K_aff=3, seed=42, dirichlet=1, prob_ratio=0.1):
        self.random_state = np.random.RandomState(seed) 

        self.probability_ratio = prob_ratio         # ratio between probability of in-community and out-community connections
        self.dirichlet = dirichlet                  # controls how strongly mixed each community memberships is on average

        self.N = N
        self.K_aff = K_aff
        
        self.initialize_latent_variables()


    def initialize_latent_variables(self):
        self.membership_vector()
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

    '''
        We consider a symmetric setting, i.e. there is only one community membership vector u. 
        Additionally, all nodes have a mixed community membership, i.e. no hard membership vectors.
    '''
    def membership_vector(self):
        u = self.random_state.dirichlet(self.dirichlet*np.ones(self.K_aff), size=self.N)
       
        # sort u based on hard memberships  
        hard_memberships_u = np.argmax(u, axis=1)
        self.u = np.array([x for x,_ in sorted(zip(u, hard_memberships_u), key=lambda pair: pair[1])])
        

    '''
        create affinity matrix with assortative community structure
    '''
    def affinity_matrix(self):
        
        K_aff = self.K_aff

        # ratio between low and high probability
        prob_ratio = self.probability_ratio
        high_prob = 1
        low_prob = high_prob * prob_ratio
        
        w = low_prob * np.ones((K_aff,K_aff))  
        np.fill_diagonal(w, high_prob * np.ones(K_aff))  

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
        self.lam = self.u @ self.w @ self.u.T 

    """
        create symmetric adjacency matrix before applying exposure 
    """
    def generate_A0(self, T, avg_degree, symmetric=False):
        self.adjust_for_avg_degree(avg_degree=avg_degree)
        data = self.random_state.poisson([self.lam]*T)
        """
            make A0 symmetric and set diagonal to zero
        """
         
        for t in range(T): 
            np.fill_diagonal(data[t],0)
            if symmetric:
                data[t] = np.triu(data[t],1) + np.triu(data[t], 1).T

        return data


    def apply_exposure(self, data, K_exp): 
        T, N = data.shape[0], data.shape[1]
        mu = self.random_state.dirichlet(self.dirichlet * np.ones(K_exp), size=N)
        self.mu = mu
        mu_ij = np.einsum('ik,jk->ij', mu, mu)
        Z = self.random_state.binomial(n=1, size=(T,N,N), p=mu_ij)
        """
            make Z symmetric and set diagonal to zero
        """
        for t in range(T): 
            Z[t] = np.triu(Z[t],1) + np.triu(Z[t], 1).T

        self.Z = Z
        return data*Z

