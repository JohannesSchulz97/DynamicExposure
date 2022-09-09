from argparse import ArgumentParser
from attrdict import AttrDict
from os import listdir

def submission_file(args):
    
    memory = args.memory
    cpus = args.cpus
    max_time = args.max_time
    disk = args.disk
    jobs = 0

    in_folder = '../../../../work/jschulz/dynamic_exposure/input/synth/'
    out_folder = '../../../../work/jschulz/dynamic_exposure/output/synth/'
    local_in_folder = '../../data/input/synth/'
    input_files = listdir(local_in_folder)
    networks = [file for file in input_files if "csv" in file]


    with open('./synth9.sub', mode='w') as f:
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

        for network in networks: 
            name = network.removesuffix('.csv')
            parts = name.split("_")
            K_exp = int(parts[2])
            if K_exp != 9: 
                continue
            for seed in seeds: 
                job_exp = f"arguments = src/main.py --algorithm Exp" \
                        f" --file {name}" \
                        f" --in_folder {in_folder} --out_folder {out_folder}" \
                        f" --seed {seed}"\
                        f" --K_exp {K_exp}\n"\
                        f"error = cluster_info/synth/{name}_Exp_{seed}.err\n" \
                        f"output = cluster_info/synth/{name}_Exp_{seed}.out\n" \
                        f"log = cluster_info/synth/{name}_Exp_{seed}.log\n" \
                        f"queue\n\n"
                job_noexp = f"arguments = src/main.py --algorithm NoExp" \
                        f" --file {name}" \
                        f" --in_folder {in_folder} --out_folder {out_folder}" \
                        f" --seed {seed}\n"\
                        f"error = cluster_info/synth/{name}_NoExp_{seed}.err\n" \
                        f"output = cluster_info/synth/{name}_NoExp_{seed}.out\n" \
                        f"log = cluster_info/synth/{name}_NoExp_{seed}.log\n" \
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
