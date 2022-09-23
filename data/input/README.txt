Synthetic Networks:


Fixed Hyperparameters in Synthetic Networks: 

- N = 100
- avg_degree_A0 = 100
- K_aff = 3
- dirichlet = 1
- prob_ratio = 0.1
- symmetric = True

Naming Convention Synthetic Input Data: 

- Data Files: seed_T_K-exp.csv
- Param Files: params_seed_T_K-exp.npz






Sociopatterns Folder: 


Each dataset contains: 
- a (T,N,N) weighted symmetric adjacency matrix
- a (5,T,N,N) binary mask used for 5-fold cross validation experiments
- a (T,N,N) binary symmetric adjacency matrix
- a membership vector mu in R^(N*K-exp), which contains the meta data exposure communities

Originally the membership vector was named u, as it was used as a proxy for the ground truth affinity community memberships in my master thesis.
The original unprocessed datasets and the preprocessing file can be found 
in the repository of my master thesis exposureMulti under the johannes branch in src/sociopattern/preprocessing.ipynb
