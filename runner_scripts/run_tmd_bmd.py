import subprocess
import shlex

cohort_dirs = [

    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210412c186_Fam71f2 & Fbxw15 & MMP20 & Oas1c & Pla2g2f & Prss33 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210405c185_Fam71f2 & Fbxw15 & Gpr83 & PD7-MXD2 & Prss33 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210329c184_Fam167a & Papd7TXD & PD7-MXD2 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210322c183_J01Rik & Calb1 & Cer1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210315c182_J01Rik & Ascl5 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210308c181_Apbb1ip & Clnk & Meis3 & Mmp24 & Prss21 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210301c180_Amelx & Arg2 & Ccl28 & Clnk & Slc38a8 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210222c179_Arg2 & Ccl28 & Fam227b & Mfng & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210215c178_Fam227b & Fam53a & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210208c177_Cd27 & Klhl1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210201c176_Ak7 & Babam1 & Fam122b & Ube3a & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210125c175_Adk & Babam1 & Fam120c & Fam122b & Krt72 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210118c174_Btbd18 & Klhl6 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210111c173_Btbd18 & Card11 & Rnf208 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210104c172_Fam185a & Prss55 & wt',
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
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20201005c159_Amelx & Gldc_ADSL_Siltrik1 & Rnf168 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200928c158_Babam1 & Fbxw15 & Prg4 & Spred1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200921c157ex_Kif6',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200921c157_Fam53a & Fbxw18 & Kif6 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200914c156_Meis3 & Prdm8 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200907c155_Aim2 & Apbb1ip & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200831c154ex_Bmp6',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200831c154_Bbip1 & Bmp6 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200824c153_Krt1 & Lrrc17 & Rnf208 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200817c152_Fam221a & Kbtbd3 & Prss54 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200810c151_Fam83d & Krtap9-5 & Rnf168 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200803c150_Churc1 & Fam219a & Fbxw16 & wt_done',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200727c149ex_Spred1',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200727c149_Cox1 & Fbxw20 & Spred1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200720c148_Emsy & Fbxw18 & Lhx5 & Spred1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200713c147_Fam160b2 & Pdgfd & Rffl & Tnfr60 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200706c146_Prss33 & Prss47 & Rhag & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200629c145_Lrrc17 & Prss54 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200622c144_Ddb1 & Fgf14 & GABRA5 & Glod4 & Pstpip1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200615c143_Krt33a & Krtap9-5 & Rnf148 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200608c142_Fam96b & Rnft1&2 & Tomm6 & Wdr63 & wt & Zc3h8',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200601c141_Ube2n & Churc1 & Cox1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200525c140_Fam160b2 & Mapkapk3 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200518c139ex_Sox14',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200518c139_Prss12 & Rffl & Sox14 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200511c138_Atg7 & Krtap8-1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200504c137_Atg7 & Ddb1 & Fam161b & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200427c136ex_Gabra5 & Pstpip1',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200427c136_Fam161b & GABRA5 & Pstpip1 & wt',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200420c135_GABRA5 & wt & Zc3h8 & Zfp474',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200413c134_Krt28 & Prdm8 & Tomm6 & wt & Zc3h8',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200413c091_Fam60a',
    '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20200406c133_Papd7TXD & Papd7-A2 & PD7-MXD2 & Prdm8 & & WT(C57Bl-6NCrl)'
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
