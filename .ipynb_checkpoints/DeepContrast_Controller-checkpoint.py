import re
import sqlite3
import pandas as pd
import os
import shutil
import subprocess
import numpy as np
import pydicom
import nrrd

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
    FAILED_CT_LOG = r"./failed_ct_log.txt"
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(os.path.abspath(TEMP_DIR))
    
    print("Reading DB")
    
    connection = sqlite3.connect(DB_DIR)
    BMD_tbl_Image_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Image", connection)
    BMD_tbl_Subject_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Subject", connection)
    # BMD_tbl_Test_df = pd.read_csv("./axial_scans.csv")[["StudyInstanceUID", "SeriesInstanceUID"]]   
    BMD_tbl_Test_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Test", connection)[["StudyInstanceUID", "SeriesInstanceUID"]]
    
    db = pd.merge(left=BMD_tbl_Subject_df, right=BMD_tbl_Test_df, on="StudyInstanceUID")
    db = pd.merge(left=db, right=BMD_tbl_Image_df[["SeriesInstanceUID", "SOPInstanceUID", "InstanceNumber"]], on="SeriesInstanceUID")
    db = db[["SubjectID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "InstanceNumber"]]
    db["Path"] = DICOM_DIR + db["SubjectID"] + "/" + db["SOPInstanceUID"] + ".dcm"
    db = db.groupby(["SubjectID", "StudyInstanceUID", "SeriesInstanceUID"])[["Path", "InstanceNumber"]].agg(list).reset_index()
    
    print("Starting loop")
    db["Contrast"] = np.nan
    db["Non-axial detected"] = False
    total = len(db.index)
    for i, row in db.iterrows():
        try:
            print(f"{'-'*70}| i = {i+1}/{total} |{'-'*70}")

            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)
            if len(os.listdir(TEMP_DIR)) != 0: # Safety check
                shutil.rmtree(os.path.abspath(TEMP_DIR))
                os.makedirs(TEMP_DIR) 
            if os.path.exists("./patient_prediction.csv"): # Safety check
                os.remove("./patient_prediction.csv")

            slices = []
            # Check to ensure that ct scan is axial
            for path, instance_number in zip(row["Path"], row["InstanceNumber"]):
                s = pydicom.dcmread(path)
                if check_orientation(s) != "axial":
                    print(f"Not axial: SeriesInstanceUID: {row['SeriesInstanceUID']} | InstanceNumber: {instance_number}")
                    db.at[i, "Non-axial detected"] = True
                    # break
                else:
                    s.InstanceNumber = instance_number
                    slices.append(s)
                    
            # if len(slices) < len(row["Path"]):
            if len(slices) == 0:
                print("Skipped: no slices")
                continue
                    
            slices = sorted(slices, key=lambda s: s.InstanceNumber, reverse=True)
            nrrd.write("./temp/temp.nrrd", np.stack([s.pixel_array for s in slices], axis=2))
            # input("")
            
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

            predictions_df = pd.read_csv("./patient_prediction.csv")

            db.at[i, "Contrast"] = predictions_df.iloc[0]["predictions"]
            print(f"Prediction: {predictions_df.iloc[0]['predictions']}")

            # input("")
            shutil.rmtree(os.path.abspath(TEMP_DIR))
            os.remove("./patient_prediction.csv")
        except Exception as e:
            db.at[i, "Contrast"] = -1 # Meaning it failed

            with open(FAILED_CT_LOG, "a") as f:
                f.writelines([
                    f"{'-'*200}\n", 
                    f"CT index: i={i}\n", 
                    f"Subject ID: {row['SubjectID']}\n", 
                    f"Series instance UID {row['SeriesInstanceUID']}\n", 
                    "ERROR:", f"\t{e}\n",
                ])

                
            
    db.to_csv("./test_df_with_contrast.csv")
    os.remove("./image_prediction.csv")
    print("Complete")
