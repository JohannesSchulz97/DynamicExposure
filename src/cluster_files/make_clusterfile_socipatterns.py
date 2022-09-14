from argparse import ArgumentParser
from attrdict import AttrDict
from os import listdir
import numpy as np

def submission_file(args):
    
    memory = args.memory
    cpus = args.cpus
    max_time = args.max_time
    disk = args.disk
    

    folds = [0,1,2,3,4]
    K_affs = [1,3,5,7,9,11]
    datasets = listdir('../../data/input/sociopatterns/')

    for dataset in datasets: 

        jobs = 0
        in_folder = f'../../../../work/jschulz/dynamic_exposure/input/sociopatterns/{dataset}/'
        out_folder = f'../../../../work/jschulz/dynamic_exposure/output/sociopatterns/{dataset}/'

        K_exp = np.load("../../data/input/sociopatterns/" + f"{dataset}/params.npz")["u"].shape[1]

        with open(f'./{dataset}.sub', mode='w') as f:
            header = "executable = /home/jschulz/anaconda3/bin/python \n"\
                    f"request_memory = {memory} \n"\
                    f"request_cpus = {cpus} \n"\
                    f"request_disk = {disk}\n"\
                    f"MaxTime = {max_time}\nNumRetries = 5\n"\
                    "periodic_hold = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))\n"\
                    "periodic_hold_reason = ifThenElse(JobRunCount <= $(NumRetries), \"Job runtime exceeded\", \"Job runtime exceeded, no more retries left\")\n"\
                    "periodic_hold_subcode = ifThenElse(JobRunCount <= $(NumRetries), 1, 2)\n"\
                    "periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 1) )\n"\
                    "job_machine_attrs = Machine\njob_machine_attrs_history_length = 3\n"\
                    "requirements = target.machine =!= MachineAttrMachine1 && target.machine =!= MachineAttrMachine2\n"\
                    "\n\n"
            f.write(header)

            seeds = [697752728, 4190089612, 1176914559, 3077924848, 315917623, 2544020234, 1077758578, 4071300106, 534591752, 3553386411]

            for seed in seeds: 
                for fold in folds: 
                    for K_aff in K_affs: 
                        job_exp = f"arguments = src/main_sociopatterns.py --algorithm Exp" \
                                f" --in_folder {in_folder} --out_folder {out_folder}" \
                                f" --seed {seed}"\
                                f" --K_aff {K_aff}"\
                                f" --K_exp {K_exp}"\
                                f" --fold {fold}\n"\
                                f"error = cluster_info/sociopatterns/{dataset}/Exp_{seed}_{fold}_{K_aff}.err\n" \
                                f"output = cluster_info/sociopatterns/{dataset}/Exp_{seed}_{fold}_{K_aff}.out\n" \
                                f"log = cluster_info/sociopatterns/{dataset}/Exp_{seed}_{fold}_{K_aff}.log\n" \
                                f"queue\n\n"
                        job_noexp = f"arguments = src/main_sociopatterns.py --algorithm NoExp" \
                                f" --in_folder {in_folder} --out_folder {out_folder}" \
                                f" --seed {seed}"\
                                f" --K_aff {K_aff}"\
                                f" --fold {fold}\n"\
                                f"error = cluster_info/sociopatterns/{dataset}/Exp_{seed}_{fold}_{K_aff}.err\n" \
                                f"output = cluster_info/sociopatterns/{dataset}/Exp_{seed}_{fold}_{K_aff}.out\n" \
                                f"log = cluster_info/sociopatterns/{dataset}/Exp_{seed}_{fold}_{K_aff}.log\n" \
                                f"queue\n\n"

                        f.write(job_exp)
                        f.write(job_noexp)
                        jobs += 2
        print(f'Number of jobs: {jobs}')

 
if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-m', '--memory', type=int, default=8192)
    p.add_argument('-c', '--cpus', type=int, default=1)
    p.add_argument('-d', '--disk', type=int, default=20000)
    p.add_argument('-l', '--max_time', type=int, default=5000)
    args = p.parse_args()
    submission_file(AttrDict(vars(args)))
