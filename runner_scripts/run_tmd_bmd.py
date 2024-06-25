import subprocess
import shlex

cohort_dirs = [
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201228c171_Prss21 & Prss35 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201221c170_Klhl1 & Meis3 & wt & Zc1HC1c',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201214c169_Mfng & Oas1c & Prdm8 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201207c168_Adk & Churc1 & Fbxw18 & Ube3a & wt_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201130c167_Krt72 & Prss33 & wt_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201123c166_Apbb1ip & Fbxw15 & Fgf20 & Prss47 & wt_c166_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201116c165ex_Bmp6',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201116c165_Bbip1 & Bmp6 & Fam81a & PWS & wt_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201109c164_Fam71d & Fam71f2 & Fam83d & Krtap9-5 & wt_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201102c163_Camk4 & Emsy & Fam185a & wt_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201026c162_Nfam1 & Prss57 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201019c161_Fam129a & Fam219a & SMIM24-2M-X & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201012c160_PD7-MXD2 & Spred1 & wt',

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
