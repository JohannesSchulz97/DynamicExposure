from argparse import ArgumentParser
import numpy as np
from attrdict import AttrDict
import sys

sys.path.append('../')
sys.path.append('./config')


from tools import *
import functools
print = functools.partial(print, flush=True)

from inference import *



def main(args):
    in_folder = args.in_folder
    out_folder = args.out_folder
    
    K_aff = args.K_aff
    K_exp = args.K_exp
    alg = args.algorithm
    exp = True if alg == 'Exp' else False
    seed = args.seed 
    fold = args.fold
    data = read_data(in_folder, "data")
    file = f"{alg}_{seed}_{fold}_{K_aff}"
    

    if fold != None: 
        masks = np.load(in_folder + "mask.npz")["mask"]
        mask = masks[fold]
        data =  data * (1-mask)
    
    print(f"Running {args.algorithm} Inference with test-fold={fold}, seed={seed}, K_aff={K_aff} and K_exp={K_exp}")
    if args.prep: 
        data = np.round(np.sqrt(data))


    losses, params = fit(data, K_aff=K_aff, K_exp=K_exp, seed=seed, exp=exp)
    if exp:        
        [mu_est, Q, u_est, v_est, w_est] = params 
        [losses, theta_errors, exp_errors] = losses
        np.savez(out_folder + "params_" + file, mu=mu_est, Q=Q, u=u_est, v=v_est, w=w_est)
        np.savez(out_folder + "training_" + file, losses=losses, theta_errors=theta_errors, exp_errors=exp_errors)
    else: 
        [losses, theta_errors] = losses
        [u_est, v_est, w_est] = params
        np.savez(out_folder + "params_" + file, u=u_est, v=v_est, w=w_est)
        np.savez(out_folder + "training_" + file, losses=losses, theta_errors=theta_errors)






if __name__ == '__main__':

    p = ArgumentParser()
    
    p.add_argument('-a', '--algorithm', type = str, choices = ['Exp', 'NoExp'], default = 'Exp',
                   help = 'Inference method to use, among exposure(Exp) and no exposure(NoExp)'
                          'Default = "Exp"')
    p.add_argument('--K_aff', type = int, default = 3, help = 'Number of affinity communities to infer.')
    p.add_argument('--K_exp', type = int, default = 3, help = 'Number of exposure communities to infer.')
    p.add_argument('-i', '--in_folder', type = str, default = '../data/input/synth/', help = 'Folder path of the input network.')
    p.add_argument('-o', '--out_folder', type = str, default = '../data/output/synth/', help = 'Path to the folder in which inference results are stored.')
    p.add_argument('-s', '--seed', type=int, default=42)
    p.add_argument('-f', '--fold', default=None, type = int, choices = [0,1,2,3,4], help='which fold to mask as a test set in the case of 5-fold cross-validation')
    p.add_argument('--prep', default=1, type=int, help='tells you if preprocessing in form of taking the square root of the data should be applied')
    args = AttrDict(vars(p.parse_args()))
    main(args)







