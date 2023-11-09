
from tmd_bmd import *



#####basic configs
NETWORK_FOLDER = Path("/home/varhai/PRIMUS/data/83_BIOINFORMATICS/Julia/Python")
cohort_folder = '/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/' \
                'Screen20210419c187_MMP20 & Pla2g2f & Ror1 & Tekt1 & wt'


############load model
json_file_head_slice = open('NN_model_classify_head_slice1.json', 'r')
loaded_model_json_head_slice = json_file_head_slice.read()
json_file_head_slice.close()

loaded_model_head_slice = model_from_json(loaded_model_json_head_slice)
loaded_model_head_slice.load_weights(NETWORK_FOLDER/"NN_model_classify_head_slice1.h5")
print("Model for head slice classification was loadded!")

json_file_masking = open(NETWORK_FOLDER/'NN_model_masking_final.json', 'r')
loaded_model_json_masking = json_file_masking.read()
json_file_masking.close()
loaded_model_masking = model_from_json(loaded_model_json_masking)
# load weights into new model
loaded_model_masking.load_weights(NETWORK_FOLDER/"NN_model_masking_final.h5")
print("Model for masking was loadded!")


json_file_head_slice_1176 = open(NETWORK_FOLDER/'NN_model_classify_head_1176_retrained.json', 'r')
loaded_model_json_head_slice_1176 = json_file_head_slice_1176.read()
json_file_head_slice_1176.close()

loaded_model_head_slice_1176 = model_from_json(loaded_model_json_head_slice_1176)
loaded_model_head_slice_1176.load_weights(NETWORK_FOLDER/"NN_model_classify_head_1176_retrained.h5")
print("Model for head slice 1176 classification was loadded!")

json_file_masking_1176 = open(NETWORK_FOLDER/'NN_model_masking_1176.json', 'r')
loaded_model_json_masking_1176 = json_file_masking_1176.read()
json_file_masking_1176.close()
loaded_model_masking_1176 = model_from_json(loaded_model_json_masking_1176)
# load weights into new model
loaded_model_masking_1176.load_weights(NETWORK_FOLDER/"NN_model_masking_1176.h5")
print("Model for masking 1176 was loadded!")

json_file_holder_masking_1176 = open(NETWORK_FOLDER/'NN_model_masking_right_holder_1176.json', 'r')
loaded_model_json_holder_masking_1176 = json_file_holder_masking_1176.read()
json_file_holder_masking_1176.close()
loaded_model_holder_masking_1176 = model_from_json(loaded_model_json_holder_masking_1176)
# load weights into new model
loaded_model_holder_masking_1176.load_weights(NETWORK_FOLDER/"NN_model_masking_right_holder_1176.h5")
print("Model for holder masking was loadded!")

AUTO = tf.data.experimental.AUTOTUNE

# Configuration
IMAGE_SIZE = [256, 256]
EPOCHS = 40
SEED = 777
BATCH_SIZE = 16


def creat_dataframe(img_path):
    os.chdir(img_path)
    thisdir = os.getcwd()
    im_paths = []
    im_ids = []
    for r, d, f in os.walk(thisdir):
        for file in f:
            if file.endswith('.bmp'):
                path = os.path.join(r,file)
                im_paths.append(path)
                im_id = file.split(".")[0]
                im_ids.append(im_id)

    df = pd.DataFrame(data={"id": im_ids, "img_path": im_paths})
    return df


def read_image_and_mask(image_path, mask_path=None, resize=IMAGE_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image,channels=3)
    image = tf.image.resize(image,IMAGE_SIZE)
    image = tf.cast(image,dtype=tf.float32) / 255.
    return image


def get_validation_dataset(df):
    validation_dataset = tf.data.Dataset.from_tensor_slices(df["img_path"].values)
    validation_dataset = validation_dataset.map(read_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    return validation_dataset


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas



#script

check_if_useful = lambda file_name : \
    file_name.endswith('body') or file_name.endswith('fantom1') \
    or file_name.endswith('fantom2') \
    or file_name.endswith('RecDXA') or file_name.endswith('mask') \
    or file_name.endswith('Morphology') or file_name.endswith('head') \
    or file_name.endswith('mask_edited') or file_name.endswith('predictions')

#####cohort folder

cohort_number = cohort_folder.split('c')[2] # Change here in case your username does not have a 'c'
cohort_number = cohort_number.split('_')[0]
cohort_number = 'c'+ cohort_number
print('Cohort folder:',cohort_folder)
print('Cohort number:',cohort_number)


MICE_IDS = [x for x in Path(cohort_folder).iterdir() if not check_if_useful(x.name)]


rule all:
    input:
        expand("{cf}/csv_files_check_new_old/{mice}",cf = cohort_folder,mice=MICE_IDS)

#
rule detect_folders:
    output: "{cf}/csv_files_check_new_old".format(cf= cohort_folder),
            expand("{cf}/csv_files_check_new_old/{mice}",cf = cohort_folder,mice=MICE_IDS)
    threads: 1

    run:
        new_old_scanner_check(cohort_folder,cohort_folder)


df_cohort = None

rule merge_folders_info:
    input: expand("{cf}/csv_files_check_new_old/{mice}",cf = cohort_folder,mice=MICE_IDS)
    output: cohort_folder + '/' + "Check_1278_1176_died.csv"
    run:

        all_files = glob.glob(os.path.join(cohort_folder + '/' + 'csv_files_check_new_old',"*.csv"))
        ##create one dataframe with all csv files
        df_from_each_file = (pd.read_csv(f,sep=',') for f in all_files)
        df_cohort = pd.concat(df_from_each_file,ignore_index=True)
        #create one csv file
        df_cohort.to_csv(cohort_folder + '/' + "Check_1278_1176_died.csv",index=False)

rule detect_fantoms:

