import subprocess
import shlex

cohort_dirs = [
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210104c172_Fam185a & Prss55 & wt'
]


for coh in cohort_dirs:
    for i in range(4):
        try:
            print(coh)

            cmd = f"python /home//varhai/PRIMUS//data//83_BIOINFORMATICS/Igor//scripts/fancy_tmd_bmd//tmd_bmd_run.py '{coh}'"
            print(cmd)
            cmd = shlex.split(cmd)
            print(cmd)
            subprocess.call(cmd)
            break
        except:
            print("WAS Exception in running")
