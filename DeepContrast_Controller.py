import re
import sqlite3
import pandas as pd
import os
import enum
import shutil
import subprocess
import numpy as np


class idTypes(enum):
    # Subject tbl
    SUBJECT_ID = 0
    STUDY_INSTANCE_UID = 1

    # Test tbl
    SERIES_INSTANCE_UID = 2

    # Image tbl
    SOP_INSTANCE_UID = 3

    def __str__(self) -> str:
        idTypesToStr = {
            idTypes.SUBJECT_ID.value: "SubjectID",
            idTypes.STUDY_INSTANCE_UID.value: "StudyInstanceUID",
            idTypes.SERIES_INSTANCE_UID.value: "SeriesInstanceUID",
            idTypes.SOP_INSTANCE_UID.value: "SOPInstanceUID",
        }

        return idTypesToStr[self.value]


class HeadCTData:
    def __init__(self, dbDir, dicomDir) -> None:
        self.dicom_dir = dicomDir

        connection = sqlite3.connect(dbDir)
        self.BMD_tbl_Image_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Image", connection)
        self.BMD_tbl_Subject_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Subject", connection)
        self.BMD_tbl_Test_df = pd.read_sql_query("SELECT * FROM BMD_tbl_Test", connection)

    def convert(self, fromType, fromVal, toType):
        if fromType is idTypes.SUBJECT_ID:
            fromType = idTypes.STUDY_INSTANCE_UID
            fromVal = self.BMD_tbl_Subject_df[self.BMD_tbl_Subject_df[str(idTypes.SUBJECT_ID)] == fromVal][
                str(idTypes.STUDY_INSTANCE_UID)
            ][0]

        if toType is idTypes.SOP_INSTANCE_UID:
            df = self.BMD_tbl_Image_df
        elif toType is idTypes.SERIES_INSTANCE_UID:
            df = self.BMD_tbl_Test_df
        else:
            df = self.BMD_tbl_Subject_df

        if toType is idTypes.SUBJECT_ID:
            seriesInstanceUid = self.convert(fromType, fromVal, idTypes.SERIES_INSTANCE_UID)
            return df[df[str(idTypes.SERIES_INSTANCE_UID)] == seriesInstanceUid][str(toType)][0]
        return df[df[str(fromType)] == fromVal][str(toType)][0]

    def get_image_path(self, sopInstanceUid):
        return os.path.join(
            self.dicom_dir,
            self.convert(idTypes.SOP_INSTANCE_UID, sopInstanceUid, idTypes.SUBJECT_ID),
            f"{sopInstanceUid}.dcm",
        )

    def get_test_image_paths(self, seriesInstanceUid):
        return [
            self.get_image_path(row[str(idTypes.SOP_INSTANCE_UID)])
            for _, row in self.BMD_tbl_Image_df[
                self.BMD_tbl_Image_df[str(idTypes.SERIES_INSTANCE_UID)] == seriesInstanceUid
            ].iterrows()
        ]


def convert_to_nrrd(command_args, verbose=True, path_to_log_file=None, return_bash_command=False):
    """
    Convert DICOM series to any supported file format.

    For additional details, see:
    https://plastimatch.org/plastimatch.html#plastimatch-convert

    Args:
        command_args: dictionary of the arguments parsable by 'plastimatch convert' that you want to use

        (GENERAL)
        path_to_log_file: path to file where stdout and stderr from the processing should be logged
        return_bash_command: return the executed command together with the exit status

    """

    bash_command = list()
    bash_command += ["plastimatch", "convert"]

    for key, val in command_args.items():
        bash_command += ["--%s" % (key), val]

    if verbose:
        print("\nRunning 'plastimatch convert' with the specified arguments:")
        for key, val in command_args.items():
            print("  --%s" % (key), val)

    try:
        if path_to_log_file:
            with open(path_to_log_file, "a") as log_file:
                bash_exit_status = subprocess.run(
                    bash_command, stdout=log_file, stderr=log_file, check=True, shell=True
                )
        else:
            # if no log file is specified, output to the default stdout and stderr
            bash_exit_status = subprocess.run(bash_command, capture_output=True, check=True, shell=True)
            print(bash_exit_status.stdout)
            print(bash_exit_status.stderr)

            if verbose:
                print("... Done.")

    except Exception as e:
        # if the process exits with a non-zero exit code, a CalledProcessError exception will be raised
        # attributes of that exception hold the arguments, the exit code, and stdout and stderr if they were captured
        # For details, see: https://docs.python.org/3/library/subprocess.html#subprocess.run

        # FIXME: return exception?
        print(e)

    if return_bash_command:
        return bash_command


if __name__ == "__main__":
    TEMP_DIR = r"./temp"
    DB_DIR = r"/nfs/MBSI/Osteoporosis/Head-CT/DB/BoneDensity_2023-01-26_final_vacuumed.db"
    DICOM_DIR = r"/nfs/MBSI/Osteoporosis/Head-CT/DICOM_extract"

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    db = HeadCTData(DB_DIR, DICOM_DIR)

    for i, row in db.BMD_tbl_Test_df.iterrows():
        seriesInstanceUid = row[str(idTypes.SERIES_INSTANCE_UID)]
        
        paths = db.get_test_image_paths(seriesInstanceUid)
        for path in paths:
            shutil.copy(path, TEMP_DIR)

        convert_to_nrrd(
            {
                "input": str(os.path.abspath(TEMP_DIR)),
                "output-img": str(os.path.abspath(os.path.join(TEMP_DIR, f"temp-{seriesInstanceUid}.nrrd"))),
                "algorithm": "itk",
            }
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
        )
    )

    predictionsDf = pd.read_csv("./patient_prediction.csv")
    db.BMD_tbl_Test_df["Contrast"] = np.nan
    for _, row in predictionsDf.iterrows():
        seriesInstanceUid = re.search("temp-(.*).nnrd", row["pat_id"]).group(1)
        db.BMD_tbl_Test_df[db.BMD_tbl_Test_df[str(idTypes.SERIES_INSTANCE_UID)]["Contrast"] == seriesInstanceUid] = row["predictions"]
        
    db.BMD_tbl_Test_df.to_csv("./test_df_with_contrast.csv")
        