import subprocess
import shlex


cohort_dirs = [

    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201214c169_Mfng & Oas1c & Prdm8 & wt'

]

# cohort_dirs = ['/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210412c186_Fam71f2 & Fbxw15 & MMP20 & Oas1c & Pla2g2f & Prss33 & wt'
# ]
MALE_FACE_PERC = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_male_f_percentiles.mat"
FEMALE_FACE_PERC = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_female_f_percentiles.mat"
coh_file = r"/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Igor/workdir/wt_cohorts_pts.pkl"
#cohort_dirs = [ '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20231120c315_Abcg5 & Gabrg1 & wt' ]

for coh in cohort_dirs:
    for i in range(4):
        try:
            print(coh)

            cmd = f"snakemake -s '/home//varhai/PRIMUS//data//83_BIOINFORMATICS/Igor//scripts/snakemake_workflows//mouse_skull_analysis.smk'  --cores {20-3*i} --config cohort_dir='{coh}'"
            cmd+= f" slicer_home=Slicer male_perc={MALE_FACE_PERC} female_perc={FEMALE_FACE_PERC} cohort_pickle={coh_file}"
            subprocess.call( cmd ,shell=True)
        except:
            print("WAS Exception in running")