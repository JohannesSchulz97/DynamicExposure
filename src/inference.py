import numpy as np
from attrdict import AttrDict
import timeit
from tools import *

from scipy.stats import poisson
from scipy.optimize import root

CONFIG = {   
            'dirichlet': 0.5,
            'tol1': 1e-5,                       # tolerance for parameter theta update errors
            'tol2': 1e-6,                        # tolerance for mu and Q update errors
            'tol3': 1e-7,                        # tolerance for convergence rate in posterior
        }

config = AttrDict(CONFIG)


def control_well_definedness(x):
    if len(x.shape) == 0: 
        if np.isnan(x) or np.isinf(x): 
            return 0
        return x
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x

def initialize_latent_variables(N, K_aff, K_exp=None, seed=42, exp=True): 
    random_state = np.random.RandomState(seed=seed)
    dir = config.dirichlet

    u = random_state.dirichlet(dir * np.ones(K_aff), size=N)
    v = random_state.dirichlet(dir * np.ones(K_aff), size=N)
    w = random_state.uniform(0,1, size=(K_aff,K_aff))
    w = np.tril(w) + np.tril(w, -1).T
    np.fill_diagonal(w, w.diagonal()*10)
    if exp==False: 
        return u,v,w
    mu = random_state.dirichlet(dir * np.ones(K_exp), size=N)
    return u,v,w,mu

def posterior(data, Q, lam, mu, eps=1e-5, symmetric=True, exp=True):
    T, N = data.shape[0], data.shape[1]
    if not exp: 
        return (data * np.log(lam+eps)- T*lam).sum()

    mu_ij = np.einsum('ik,jk->ij', mu, mu)

    posterior = 0
    # add the entropy of Q
    entropy = control_well_definedness(Q*np.log(1/Q) + (1-Q)*np.log(1/(1-Q)))

    
    if symmetric: 
        posterior += np.tril(np.sum(entropy, axis=0), -1).sum()
    else: 
        posterior += entropy.sum()
    
    # add part coming from prior on Z 
    prior = control_well_definedness(np.log(mu_ij) * Q.sum(axis=0) + np.log(1-mu_ij) * (T - Q.sum(axis=0)))
    posterior += np.tril(prior, -1).sum()

    # add part coming from likelihood
    likelihood = Q * (data * np.log(lam+eps) + lam)
    if symmetric: 
        posterior += np.tril(likelihood.sum(axis=0), -1).sum()
    else: 
        posterior += likelihood.sum()

    # 'normalize' posterior to so that it lies in the same range for different N and T
    posterior /= T*N**2
    return posterior




"""
    inference functions with exposure and without exposure
"""
def fit(data, K_aff, K_exp, seed=42, symmetric=False, triangular_w=False, iter1=1000, iter2=10, iter3=5, exp=True, verbose=False, 
        true_u=None, true_v=None, true_w=None, true_mu=None, true_Z=None,               # these are the true values, they will remain unchanged during inference                     
        initial_u=None, initial_v=None, initial_w=None, initial_mu=None, initial_Q=None):                 # these will be used initially, but updated over time
    
    if not exp: 
        return fit_noexp(data, K_aff, seed=seed, symmetric=symmetric, triangular_w=triangular_w, iter1=20000)
    T, N = data.shape[0], data.shape[1]
    u,v,w,mu = initialize_latent_variables(N, K_aff, K_exp, seed=seed)

    if initial_u is not None: 
        u=initial_u
    if initial_v is not None: 
        v=initial_v
    if initial_w is not None: 
        w=initial_w
    if initial_mu is not None: 
        mu=initial_mu
    if initial_Q is not None:  
        Q=initial_Q

    if true_u is not None: 
        u=true_u
    if true_v is not None: 
        v=true_v
    if true_w is not None: 
        w=true_w
    if true_mu is not None: 
        mu=true_mu
    if true_Z is not None: 
        Q=true_Z

    lam = np.einsum('iq, jq -> ij', np.einsum('ik, kq -> iq', u, w), v) 

    changing_u = (true_u is None)
    changing_v = (true_v is None)
    changing_w = (true_w is None)
    changing_mu = (true_mu is None)
    changing_Q = (true_Z is None)

    if changing_Q: 
        Q = update_Q(data, lam, mu, symmetric=symmetric)

    update_theta = changing_u or changing_v or changing_w
    changing_exp = changing_Q or changing_mu
    u_error, v_error, w_error, mu_error, Q_error = 0,0,0,0,0
    
    probabilities = []
    theta_errors = []
    exp_errors = []
    start_time = timeit.default_timer()

    iterations = 0
    print(f"running Exp inference for {iter1*(iter2+iter3)} iterations: \n")
    for _ in range(iter1): 

        '''
            update u, v and w for iter2 iterations
        '''
        QA_sum = (Q*data).sum(axis=0)
        Q_sum = Q.sum(axis=0)
        for _ in range(iter2):
            if update_theta: 
                
                if changing_u: 
                    init_time = timeit.default_timer()
                    rho = update_rho(u,v,w)
                    '''
                    if symmetric
                    assert(rho.transpose((1,0,3,2)))
                    '''
                    u_new = update_u(QA_sum, Q_sum, rho, v, w)
                    if verbose:
                        print(f"updating u took {timeit.default_timer() - init_time} seconds")
                    u_error = np.abs(u - u_new).sum() / u.size
                    u = u_new
                if changing_v: 
                    if symmetric: 
                        v = u.copy()
                        v_error = u_error
                    else:
                        init_time = timeit.default_timer()
                        rho = update_rho(u,v,w)
                        v_new = update_v(QA_sum, Q_sum, rho, u, w)
                        if verbose:
                            print(f"updating v took {timeit.default_timer() - init_time} seconds")
                        v_error = np.abs(v - v_new).sum() / v.size
                        v = v_new
                if changing_w: 
                    init_time = timeit.default_timer()
                    rho = update_rho(u,v,w)
                    w_new = update_w(QA_sum, Q_sum, rho, u, v)
                    if symmetric and triangular_w: 
                        w_new = np.tril(w_new)
                    if verbose:
                        print(f"updating w took {timeit.default_timer() - init_time} seconds")
                    w_error = np.abs(w - w_new).sum() / w.size
                    w = w_new
                
                theta_error = (u_error + v_error + w_error) / (changing_u + changing_v + changing_w)
                theta_errors.append(theta_error)
                iterations += 1
                if (iterations % 10) == 0:
                    init_time = timeit.default_timer()
                    lam = np.einsum('iq, jq -> ij', np.einsum('ik, kq -> iq', u, w), v)
                    p = posterior(data, Q, lam, mu, symmetric=symmetric)
                    probabilities.append(p)
                    if verbose:
                        print(f"updating the posterior took {timeit.default_timer() - init_time} seconds")
                if theta_error < config.tol1: 
                    if verbose:
                        print("update size on u,v and w smaller than tolerance")
                    break

        '''
            update mu and Q for iter3 iterations
        '''
        for _ in range(iter3): 
            if changing_exp: 
                if changing_mu: 
                    init_time = timeit.default_timer()
                    mu_new = update_mu(Q, mu)
                    if verbose:
                        print(f"updating mu took {timeit.default_timer() - init_time} seconds")
                    mu_error = np.abs(mu - mu_new).sum() / mu.size
                    mu = mu_new
                if changing_Q: 
                    init_time = timeit.default_timer()
                    Q_new = update_Q(data, lam, mu, symmetric=symmetric)
                    if verbose:
                        print(f"updating Q took {timeit.default_timer() - init_time} seconds")
                    Q_error = np.abs(Q - Q_new).sum() / Q.size
                    Q = Q_new
                exp_error = (mu_error + Q_error) / (changing_mu + changing_Q)
                exp_errors.append(exp_error)
                iterations += 1
                if (iterations % 10) == 0:
                    init_time = timeit.default_timer()
                    lam = np.einsum('iq, jq -> ij', np.einsum('ik, kq -> iq', u, w), v)
                    p = posterior(data, Q, lam, mu, symmetric=symmetric)
                    probabilities.append(p)
                    if verbose:
                        print(f"updating the posterior took {timeit.default_timer() - init_time} seconds")
                if exp_error < config.tol2: 
                    if verbose:
                        print("update size on mu and Q smaller than tolerance")
                    break

        '''
            terminate automatically after 15000 iterations
        '''
        if iterations > 15000: 
            break

        '''
            check convergence based on update size of parameters and convergence in loss function
        '''
        parameter_criterium = True
        if update_theta and (theta_errors[-1] > config.tol1): 
            parameter_criterium = False
        if changing_Q and (exp_errors[-1] > config.tol2):
            parameter_criterium = False
        
        """
            we require a minimum of 1000 iterations and a convergence in loss function
        """
        loss_criterium = False
        if (iterations >=1000) and (np.abs(probabilities[-1] - probabilities[-2]) < config.tol3):
            loss_criterium = True
        
        if parameter_criterium and loss_criterium: 
            print("Convergence Criterium satisfied")
            break


    elapsed = timeit.default_timer() - start_time

    print(f"Exposure inference with T={T}, N={N}, K={K_aff} and {iterations} iterations took {elapsed} seconds\n")
    
    for t in range(T): 
        np.fill_diagonal(Q[t], 0)
    return [probabilities, theta_errors, exp_errors], [mu,Q,u,v,w]



def fit_noexp(data, K, seed=42, symmetric=False, triangular_w=False, iter1=5000, 
                true_u=None, true_v=None, true_w=None, 
                initial_u=None, initial_v=None, initial_w=None):
    true_T = data.shape[0]
    data = np.expand_dims(data.sum(axis=0), 0)
    T, N = data.shape[0], data.shape[1]
    u,v,w = initialize_latent_variables(N, K, seed=seed, exp=False)
    
    if initial_u is not None: 
        u=initial_u
    if initial_v is not None: 
        v=initial_v
    if initial_w is not None: 
        w=initial_w

    if true_u is not None: 
        u=true_u
    if true_v is not None: 
        v=true_v
    if true_w is not None: 
        w=true_w

    changing_u = (true_u is None)
    changing_v = (true_v is None)
    changing_w = (true_w is None)
    u_error, v_error, w_error = 0,0,0

    if symmetric: 
        v = u.copy()
    probabilities = []
    theta_errors = []
    start_time = timeit.default_timer()
    iterations = 0
    print(f"running NoExp inference for {iter1} iterations: \n")
    for i in range(iter1): 
        
        # update u
        if changing_u: 
            rho = update_rho(u,v,w)
            numerator = np.einsum('tmj,mjnq->mn', data, rho)
            denominator= T * np.einsum('jq,nq->n', v, w)
            u_new = control_well_definedness(numerator/denominator)
            u_error = np.abs(u - u_new).sum() / u.size
            u = u_new
     
        # update v
        if changing_v: 
            if symmetric:
                v = u.copy()
                v_error = u_error
            else: 
                rho = update_rho(u,v,w)
                numerator = np.einsum('tim,imkn->mn', data, rho)
                denominator = T * np.einsum('ik,kn->n', u, w)
                v_new = control_well_definedness(numerator/denominator)
                v_error = np.abs(v - v_new).sum() / v.size
                v = v_new

        if changing_w:
            # update w
            rho = update_rho(u,v,w)
            numerator = np.einsum('tij,ijmn->mn', data, rho)
            denominator = T * np.einsum('im,jn->mn', u, v)
            w_new = control_well_definedness(numerator/denominator)
            if triangular_w: 
                w_new = np.tril(w_new)
            w_error = np.abs(w - w_new).sum() / w.size
            w = w_new

        '''
        print(f"iteration {i}:\n u={u[0]},\n v={v[0]}, \n w={w}\n\n")
        if i == 2: 
            return
        '''    

        theta_error = (u_error + v_error + w_error) / 3
        theta_errors.append(theta_error)
        
        iterations += 1
        if (iterations % 10) == 0:
            lam = np.einsum('iq, jq -> ij', np.einsum('ik, kq -> iq', u, w), v)         
            p = posterior(data, None, lam, None, exp=False)
            probabilities.append(p)
            
        """
            we require a minimum of 1000 iterations and a convergence in loss function
        """
        if (iterations >=1000) and (np.abs(probabilities[-1] - probabilities[-2]) < config.tol3) and (theta_error < config.tol1): 
            print("Convergence Criterium satisfied")
            break

        """
            terminate automatically after 20000 iterations
        """
        if iterations > 10000: 
            break

    elapsed = timeit.default_timer() - start_time

    print(f"No exposure inference with T={true_T}, N={N}, K={K} and {iterations} iterations took {elapsed} seconds\n")  
    return [probabilities, theta_errors], [u,v,w]



'''
    parameter update functions for mu, Q, rho and theta
'''

def mu_ik_func(mu_ik, i, k, mu, num1, num2, Mu_ij, eps=1e-5):
    if mu_ik-eps < 0 or mu_ik+eps > 1:
        return 1000000

    denom1 = mu_ik * mu[:,k] + Mu_ij
    denom2 = 1-denom1

    A = mu[:,k] * (control_well_definedness(num1/denom1) - control_well_definedness(num2/denom2))
    return np.delete(A, i).sum()

def update_mu(Q, mu):
    T, N, K = Q.shape[0], mu.shape[0], mu.shape[1]
    num1 = Q.sum(axis=0)
    num2 = T-num1
    for k in range(K): 
        Mu_ij = np.einsum('il,jl->ij', np.delete(mu, k, axis=1), np.delete(mu,k, axis=1))
        for i in range(N):
            root_result = root(mu_ik_func, mu[i, k], args=(i,k,mu,num1,num2,Mu_ij[:,i]))
            mu[i, k] = root_result.x            
    return mu

def update_Q(data, lam, mu, symmetric=False): 
    T = data.shape[0]
    mu_ij = np.einsum('ik,jk->ij', mu, mu)
    data_T = data.transpose((0,2,1))

    positive = np.expand_dims(mu_ij, 0) * poisson.pmf(data,[lam]*T) #* poisson.pmf(data_T,[lam.T]*T)
    negative = np.expand_dims(1-mu_ij, 0) * (data == 0) #* (data_T == 0)

    if not symmetric:
        positive *= poisson.pmf(data_T,[lam.T]*T)
        negative *= (data_T == 0)

    return control_well_definedness(positive / (positive + negative))

def update_rho(u,v,w):
    numerator = np.einsum('ik,jq,kq->ijkq', u, v, w)
    denominator = np.expand_dims(np.einsum('ijkq->ij', numerator), (2,3))
    return control_well_definedness(numerator/denominator)

def update_u(QA_sum, Q_sum, rho, v, w):
    numerator = np.einsum('mjn,mj->mn', rho.sum(axis=3), QA_sum)
    denominator = np.einsum('jq,nq,mj->mn', v, w, Q_sum)
    return control_well_definedness(numerator/denominator)

def update_v(QA_sum, Q_sum, rho, u, w):
    numerator = np.einsum('imn,im->mn', rho.sum(axis=2), QA_sum)
    denominator = np.einsum('ik,kn,im->mn', u, w, Q_sum)
    return control_well_definedness(numerator/denominator)

def update_w(QA_sum, Q_sum, rho, u, v):
    numerator = np.einsum('ijmn,ij->mn', rho, QA_sum)
    denominator = np.einsum('im,jn,ij->mn', u, v, Q_sum)
    return control_well_definedness(numerator/denominator)

