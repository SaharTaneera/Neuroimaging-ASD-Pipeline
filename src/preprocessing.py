import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from nilearn import datasets, input_data, connectome

def load_abide_dataset(data_dir="./ABIDE_data", n_subjects=100):
    os.makedirs(data_dir, exist_ok=True)
    print("Fetching ABIDE PCP dataset...")
    datasets.fetch_abide_pcp(
        data_dir=data_dir, 
        pipeline='cpac', 
        derivatives=['func_preproc'],
        n_subjects=n_subjects
    )
    direct_dir = os.path.join(data_dir, 'ABIDE_pcp', 'cpac', 'nofilt_noglobal')
    labels_file = os.path.join(data_dir, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
    return direct_dir, labels_file

def extract_features(direct_dir, labels_file, max_files=100):
    file_list = sorted([f for f in os.listdir(direct_dir) if f.endswith('.nii.gz')])
    files_to_load = file_list[:max_files]

    labels_df = pd.read_csv(labels_file)
    file_id_to_label = dict(zip(labels_df['SUB_ID'], labels_df['DX_GROUP']))

    print("Loading Harvard-Oxford Atlas...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    atlas_img = nib.load(atlas.filename)
    
    masker = input_data.NiftiLabelsMasker(
        labels_img=atlas_img, 
        standardize=True,
        memory='nilearn_cache', 
        verbose=0
    )
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')

    features = []
    labels_list = []

    print("Extracting time series and computing connectivity matrices...")
    for file_name in tqdm(files_to_load, desc="Processing fMRI Scans"):
        file_path = os.path.join(direct_dir, file_name)
        img = nib.load(file_path)

        time_series = masker.fit_transform(img)
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        connectivity_vector = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        features.append(connectivity_vector)

        file_id = None
        parts = file_name.split('_')
        for part in parts:
            if len(part) == 7 and part.isdigit():
                file_id = int(part)
                break

        if file_id in file_id_to_label:
            labels_list.append(file_id_to_label[file_id])
        else:
            match = labels_df[labels_df['FILE_ID'] == file_name.replace('.nii.gz', '')]
            if not match.empty:
                labels_list.append(match['DX_GROUP'].values[0])
            else:
                features.pop()

    return np.array(features), np.array(labels_list)
