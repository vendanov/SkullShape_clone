


import pandas as pd
from pathlib import Path
import sys
import dataclasses
import shutil

csv_filtered_mice = pd.read_csv(sys.argv[1])
all_cohorts_dir = sys.argv[2]
output_dir = sys.argv[3]

# read all mouse
@dataclasses.dataclass
class Mouse:
    path_dir: str
    mouse_id : str
    gender : str


all_cohorts= {}


# script reads all subfolders in all_cohorts_dir and creates a dict with cohort number as key and list of subfolders as value


for path in Path(all_cohorts_dir).iterdir():
    if path.is_dir() and path.name.startswith("20"):
        # iterate over subfolders
        for subdir in path.iterdir():
            if subdir.is_dir():
                scr_substr = subdir.name.split("_")[0]

                cohort_num_str = scr_substr.split('c')[-1]

                # if it is not converts to int it is old cohort. dont use those
                try:
                    cohort_num = int(cohort_num_str)
                except ValueError:
                    cohort_num = 0

                print(cohort_num)

                if cohort_num != 0:
                    cstr = 'c' + cohort_num_str
                    print(cstr)
                    if cstr in all_cohorts:
                        all_cohorts[cstr].append(subdir)
                    else:
                        all_cohorts[cstr] = [subdir]



# now copy mouse heads to result dir.
all_mouses = []


for id in csv_filtered_mice.index:

    mouse_row = csv_filtered_mice.loc[id]

    coh_id = mouse_row["COHORT_ID"]
    if coh_id not in all_cohorts.keys():
        print("Error: cohort {} not found".format(coh_id))
        continue
    else:
        coh_foldes = all_cohorts[coh_id]
        if len(coh_foldes) >1 :
            print("More than one cohort folder found")

        # find mouse with ID

        for folder in coh_foldes:
            if (folder/"skull_shape_workdir").exists():
                for subdir_ in(folder/"skull_shape_workdir").iterdir():
                    if mouse_row['ID'] in subdir_.name:
                        print(subdir_)
                        if subdir_.is_dir() and (subdir_/ "ct_image.nii.gz").exists():
                            mouse = Mouse(subdir_, mouse_row['ID'], mouse_row['GENDER'])
                            all_mouses.append(mouse)
            else:
                print("skull_shape_workdir not found. list of subfolers{}".format(list((folder).iterdir())))






    print(mouse_row)





print(len(all_mouses))

for mouse in all_mouses:
    print(mouse)

    # make out dir with mouse_Id as name

    Path.mkdir(Path(output_dir)/mouse.mouse_id,exist_ok=True)

    # copy file ct_image.nii.gz into created folder

    shutil.copy( str(mouse.path_dir/"ct_image.nii.gz"), str(Path(output_dir)/mouse.mouse_id/"ct_image.nii.gz") )














