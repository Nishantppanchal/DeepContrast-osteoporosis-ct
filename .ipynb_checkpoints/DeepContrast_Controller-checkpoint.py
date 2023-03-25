import re
import sqlite3
import pandas as pd
import os
from enum import Enum
import shutil
import subprocess
import numpy as np
import pydicom

class HeadCTData:
    def __init__(self, dbDir, dicomDir) -> None:
        self.dicom_dir = dicomDir

        connection = sqlite3.connect(dbDir)
        self.BMD_tbl_Image_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Image", connection)
        self.BMD_tbl_Subject_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Subject", connection)
        self.BMD_tbl_Test_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Test", connection)

    def get_image_path(self, subject_id, sop_instance_uid):
        return os.path.join(
            self.dicom_dir,
            subject_id,
            f"{sop_instance_uid}.dcm",
        )

    def get_test_image_paths(self, subject_id, series_instance_uid):
        return [
            self.get_image_path(subject_id, row["SOPInstanceUID"])
            for _, row in self.BMD_tbl_Image_df[
                self.BMD_tbl_Image_df["SeriesInstanceUID"] == series_instance_uid
            ].iterrows()
        ]
    
    def get_series(self, study_instance_uid):
        return self.BMD_tbl_Test_df[self.BMD_tbl_Test_df["StudyInstanceUID"] == study_instance_uid]
    
# Takes in DICOM image returns orientation
def check_orientation(ct_scan): 
    image_ori = ct_scan.ImageOrientationPatient
    image_y = np.array([image_ori[0], image_ori[1], image_ori[2]])
    image_x = np.array([image_ori[3], image_ori[4], image_ori[5]])
    image_z = np.cross(image_x, image_y)
    abs_image_z = abs(image_z)
    main_index = list(abs_image_z).index(max(abs_image_z))
    if main_index == 0:
        main_direction = "sagittal"
    elif main_index == 1:
        main_direction = "coronal"
    else:
        main_direction = "axial"
    return main_direction


if __name__ == "__main__":
    TEMP_DIR = r"./temp"
    DB_DIR = r"/nfs/MBSI/Osteoporosis/Head-CT/DB_filtered/BoneDensity_2023-03-01_10 - deid.db"
    DICOM_DIR = r"/nfs/MBSI/Osteoporosis/Head-CT/DICOM_filtered/"
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(os.path.abspath(TEMP_DIR))
    
    print("Reading DB")
    db = HeadCTData(DB_DIR, DICOM_DIR)
    
    print("Starting loop")
    db.BMD_tbl_Test_df["Contrast"] = np.nan
    for i, subject_row in db.BMD_tbl_Subject_df.iterrows():
        
        subject_id = subject_row["SubjectID"]
        study_instance_uid = subject_row["StudyInstanceUID"]
        series = db.get_series(study_instance_uid)
        
        for j, series_row in series.iterrows():
            print("-"*150)
            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)
            if len(os.listdir(TEMP_DIR)) != 0: # Safety check
                shutil.rmtree(os.path.abspath(TEMP_DIR))
                os.makedirs(TEMP_DIR)
            
            series_instance_uid = series_row["SeriesInstanceUID"]
            paths = db.get_test_image_paths(subject_id, series_instance_uid)

            # Check to ensure that ct scan is axial
            not_axial_flag = False
            for path in paths:
                if check_orientation(pydicom.dcmread(path)) != "axial":
                    not_axial_flag = True
            if not_axial_flag:
                print("Not axial")
                continue


            for path in paths:
                shutil.copy(path, os.path.abspath(TEMP_DIR))

            subprocess.call(
                (
                    "plastimatch",
                    "convert",
                    "--input", 
                    str(os.path.abspath(TEMP_DIR)),
                    "--output-img", 
                    str(os.path.abspath(os.path.join(TEMP_DIR, f"temp.nrrd"))),
                    "--algorithm", 
                    "itk",
                )
            )

            subprocess.call(
                (
                    "python",
                    "run_inference.py",
                    "--body_part",
                    "HeadNeck",
                    "--data_dir",
                    str(os.path.abspath(TEMP_DIR)),
                    "--save_csv",
                    "True",
                )
            )

            predictionsDf = pd.read_csv("./patient_prediction.csv")

            db.BMD_tbl_Test_df["Contrast"] = predictionsDf.iloc[0]["predictions"]
            print(f"Prediction {i}-{j}: {predictionsDf.iloc[0]['predictions']}")

            shutil.rmtree(os.path.abspath(TEMP_DIR))
            os.remove("./patient_prediction.csv")
            
    db.BMD_tbl_Test_df.to_csv("./test_df_with_contrast.csv")
    os.remove("./image_prediction.csv")
    print("Complete")
