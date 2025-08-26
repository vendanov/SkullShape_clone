import subprocess
import shlex
import sys
from pathlib import Path
import yaml



def change_config_path(yaml_path, new_cohort_dir):

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['cohort_dir'] = new_cohort_dir
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)


home_fold = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/"
cohort_dirs = [ x for x in Path(home_fold).iterdir() if x.is_dir() and not (x/'abnormality_automated_combined.csv').exists() ]

# filter out just the ones which between c200 -  c300
cleaned_cohort_dirs = []
for d in cohort_dirs:
    print(d.name.split('_')[0][-3:])
    try:
        if d.name.split('_')[0][-4] == 'c':
            pass
        else:
            continue
        num = int(d.name.split('_')[0][-3:])
        if 200 < num < 300:
            cleaned_cohort_dirs.append(d)
    except:
        pass

cohort_dirs = cleaned_cohort_dirs

#cohort_dirs = [ x for x in cohort_dirs if 200 < int(x.name.split('_')[0][-3:]) < 300 ]

print(cohort_dirs)
#psys.exit(0)
# cohort_dirs = ['/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210412c186_Fam71f2 & Fbxw15 & MMP20 & Oas1c & Pla2g2f & Prss33 & wt'
# ]


config_path = '/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_pipelines/smk_configs/mouse_skull.yaml'

config_path_fb = '/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_full_body/smk_scripts/mouse_skull.yaml'

conda_env = '/home/varhai/miniconda3/envs/mouseSkull'
#cohort_dirs = [ '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20231120c315_Abcg5 & Gabrg1 & wt' ]

for coh in cohort_dirs:
    print(coh)

    change_config_path(config_path, str(coh))
    change_config_path(config_path_fb, str(coh))

    # remove skull_shape_workdir

    subprocess.call(f'rm -rf "{str(coh)}/skull_shape_workdir"', shell=True, executable='/bin/bash')
    # run full body
    cmd = f"""
        source /usr/local/anaconda/etc/profile.d/conda.sh && \
        conda activate {conda_env} && \
        snakemake -s '/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_full_body/smk_scripts/mouse_full_body.smk' \
        --cores 19 --configfile {config_path_fb}
    """
    subprocess.call(cmd, shell=True, executable='/bin/bash')

    # done
    for i in range(4):
        try:

            print(coh)
            cmd = f"""
                source /usr/local/anaconda/etc/profile.d/conda.sh && \
                conda activate {conda_env} && \
                snakemake -s '/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/mouse_skull_pipelines/snakemake_workflows/mouse_skull_analysis.smk' \
                --cores {20 - 3 * i} --configfile {config_path}
            """
            subprocess.call(cmd, shell=True, executable='/bin/bash')
        except:
            print("WAS Exception in running")