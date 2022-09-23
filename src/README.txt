tests Folder: 
- contains comparison notebooks between symmetric and asymmetric inference for Exp and NoExp


Python Source Files: 

inference.py: 
- contains Exp and NoExp inference

network.py: 
- abstract Network class
- use generate_A0 and apply_exposure to generate concrete instances

tools.py: 
- contains all kind of helper functions like plotting, analysis, reading and writing data, ...

main_synth.py and main_sociopatterns.py: 
- main functions used in conjunction with make_clusterfile_{synth/sociopatterns} respectively


Notebooks: 

preprocessing.ipynb
- old preprocessing file used to generate Sociopatterns datasets
- to use, you need to download the original datasets

influence_K_exp
- analyzes the influence of K_exp on synthetic networks when keeping all other hyperparameters constant
- used to generate changing_K_exp.gif

analysis_synth.ipynb
- analysis of the experiments on synthetic data

analysis_sociopatterns.ipynb
- analysis on workplace01 dataset

debugging.ipynb
- debugging of results on synthetic experiments, especially the strange losscurves observed in the beginning