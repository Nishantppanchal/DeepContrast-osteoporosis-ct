import math
import pydicom
from pydicom.pixel_data_handlers import (
    gdcm_handler,
    pillow_handler,
    numpy_handler,
    apply_modality_lut,
)
import os
import numpy as np
from enum import Enum, auto
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from typing import List
from monai.transforms import (
    Zoom,
    RandSpatialCrop,
    RandGaussianSmooth,
    RandStdShiftIntensity,
    RandAdjustContrast,
    RandCoarseDropout,
    RandCoarseShuffle,
    HistogramNormalize,
    Spacing,
    RandRotate,
    RandFlip,
    RandZoom,
    RandAffine,
    RandGridDistortion,
    Rand2DElastic,
    RandSmoothFieldAdjustContrast,
)
import random
from itertools import combinations
from multiprocessing import Pool
import time
from functools import reduce
from pathlib import Path
import nrrd
from prediction.data_prepro import data_prepro
from prediction.model_pred import model_pred
# import pyplastimatch
import subprocess

class Plane(Enum):

    AXIAL = auto()
    CORONAL = auto()
    SAGITTAL = auto()


TRANSFORMATIONS = [
    RandSpatialCrop(roi_size=(1, 2)),
    RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), prob=1.0),
    # RandGaussianNoise, -> Rory
    # RandShiftIntensity, -> Rory
    RandStdShiftIntensity(factors=(1, 4)),
    # RandBiasField, -> Rory
    # RandScaleIntensity, -> Rory
    # NormalizeIntensity, -> Rory
    # ThresholdIntensity, -> Rory
    RandAdjustContrast(prob=1.0, gamma=1.0),
    # MaskIntensity, -> Rory
    # SavitzkyGolaySmooth, -> Rory
    # RandGaussianSharpen, -> Rory
    # RandHistogramShift, -> Rory
    # DetectEnvelope, -> Rory
    # RandGibbsNoise, -> Rory
    # RandKSpaceSpikeNoise, -> Rory
    # RandRicianNoise, -> Rory
    RandCoarseDropout(holes=10, spatial_size=(10, 10), fill_value=6),
    RandCoarseShuffle(holes=7, spatial_size=(0, 4)),
    HistogramNormalize(num_bins=256, min=0, max=255),
    # ForegroundMask, -> Rory
    Spacing(pixdim=(0, 2)),
    RandRotate(range_x=(-2, 2), prob=1.0),
    RandFlip(prob=1.0, spatial_axis=0),
    # RandAxisFlip, -> Rory
    RandZoom(prob=1.0, min_zoom=0.9, max_zoom=1.1),
    RandAffine(prob=1.0, shear_range=(0.0, 1.0)),
    RandGridDistortion(num_cells=5, prob=1.0, distort_limit=(-0.03, 0.03)),
    Rand2DElastic(spacing=(0, 0), magnitude_range=(0, 0), prob=1.0),
    RandSmoothFieldAdjustContrast(
        spatial_size=(512, 512), rand_size=(10, 100), pad=0, prob=1.0, gamma=1.0
    ),
    # RandSmoothFieldAdjustIntensity, -> Rory
    # RandomMotion, -> Rory
    # RandomGhosting, -> Rory
]
# TRANSFORMATIONS = [
#     RandSpatialCrop(roi_size=(1, 2)),
#     RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), prob=1.0),
# ]


def read_folder(files, root, cls, device):
    files.sort()
    return cls(
        root,
        [
            pydicom.dcmread(os.path.join(root, file))
            for file in files
            if file.endswith(".dcm")
        ],
        device,
    )


class CTScan:
    def __init__(self, path, slices, device) -> None:
        pydicom.config.pixel_data_handlers = [
            gdcm_handler,
            numpy_handler,
            pillow_handler,
        ]

        self.__path = path
        self.__device = device
        self.__patient_name = slices[0].PatientName

        for ct_slice in slices:
            ct_slice.decompress()

        try:
            self.__slices = sorted(slices, key=lambda s: s.SliceLocation, reverse=True)
            self.sorted = True
        except:
            self.__slices = slices
            self.sorted = False

        self.__3d_view = torch.tensor(
            np.array([s.pixel_array for s in slices], dtype=np.float32),
            requires_grad=False,
            dtype=torch.float32,
            device=self.__device,
        )

        self.__interpolated_blocks = []
        self.__interpolated = False

    @classmethod
    def read_data(cls, dir_path, device=None) -> List["CTScan"]:
        if not device:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        i = 1

        ct_scans = []
        for root, _, files in os.walk(dir_path):
            print(f"progress: {i}", end="\r")
            i += 1

            if len(files):
                files.sort()
                ct_scans.append(
                    cls(
                        root,
                        [
                            pydicom.dcmread(os.path.join(root, file))
                            for file in files
                            if file.endswith(".dcm")
                        ],
                        device,
                    )
                )

        return ct_scans

    @classmethod
    def read_data_parallel(
        cls, dir_path, device=None, num_workers=10
    ) -> List["CTScan"]:
        if not device:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        with Pool(processes=num_workers) as pool:

            p_start = time.time()
            print(f"pool creation: {p_end - p_start}")

            results = [
                pool.apply_async(read_folder, (files, root, cls, device))
                for root, _, files in os.walk(dir_path)
                if len(files)
            ]
            p_end = time.time()
            ct_scans = [res.get() for res in results]

        return ct_scans

    def get_path(self):
        return self.__path

    def get_size(self):
        return tuple(self.__3d_view.shape)

    def get_images(self, plane=Plane.AXIAL, require_grad=False) -> List[torch.Tensor]:
        if plane is Plane.AXIAL:
            return [
                self.__3d_view[i, :, :].clone().requires_grad_(require_grad)
                for i in range(self.__3d_view.size(dim=0))
            ]
        elif plane is Plane.CORONAL:
            return [
                self.__3d_view[:, :, i].clone().requires_grad_(require_grad)
                for i in range(self.__3d_view.size(dim=2))
            ]
        elif plane is Plane.SAGITTAL:
            return [
                self.__3d_view[:, i, :].clone().requires_grad_(require_grad)
                for i in range(self.__3d_view.size(dim=1))
            ]
        else:
            raise ValueError("The argument plane is invalid")

    def get_3d(self):
        return self.__3d_view

    def show(self, slice_num_axial, slice_num_coronal, slice_num_sagittal):
        # Creates a subplot
        f, axarr = plt.subplots(1, 3, figsize=(20, 20))

        axial_image = self.get_images(plane=Plane.AXIAL)[slice_num_axial].cpu().numpy()
        coronal_image = (
            self.get_images(plane=Plane.CORONAL)[slice_num_coronal].cpu().numpy()
        )
        sagittal_image = (
            self.get_images(plane=Plane.SAGITTAL)[slice_num_sagittal].cpu().numpy()
        )

        axarr[0].imshow(axial_image, cmap="gray")
        axarr[0].set_title("Axial image")

        axarr[0].add_patch(
            Rectangle(
                (slice_num_coronal, 0),
                1,
                self.__3d_view.size(dim=2),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        axarr[0].add_patch(
            Rectangle(
                (0, slice_num_sagittal),
                self.__3d_view.size(dim=1),
                1,
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
        )

        axarr[1].imshow(coronal_image, cmap="gray")
        axarr[1].set_title("Coronal image")

        axarr[1].add_patch(
            Rectangle(
                (slice_num_sagittal, 0),
                1,
                self.__3d_view.size(dim=0),
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
        )
        axarr[1].add_patch(
            Rectangle(
                (0, slice_num_axial),
                self.__3d_view.size(dim=1),
                1,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
        )

        axarr[2].imshow(sagittal_image, cmap="gray")
        axarr[2].set_title("Sagittal image")

        axarr[2].add_patch(
            Rectangle(
                (slice_num_coronal, 0),
                1,
                self.__3d_view.size(dim=0),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        axarr[2].add_patch(
            Rectangle(
                (0, slice_num_axial),
                self.__3d_view.size(dim=2),
                1,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
        )

        plt.show()

    def transform_to_hu(self):
        for i, s in enumerate(self.__slices):
            pixel_array = apply_modality_lut(s.pixel_array, s).astype(np.float16)
            s.PixelData = pixel_array.tobytes()
            self.__3d_view[i, :, :] = torch.from_numpy(pixel_array)

    def window_images(self, window_width, window_level):
        # Find the max and min HU values
        min_value = window_level - (window_width / 2)
        max_value = window_level + (window_width / 2)

        self.__3d_view = self.__3d_view.clip(min_value, max_value)

    def crop_image(self):
        # Get a matrix the define where the foreground is and where the background is
        # The foreground pixels are marked as True, while the background pixels are marked as False
        # This tensor is then converted to a numpy array
        foreground_mask = (self.__3d_view != torch.min(self.__3d_view)).cpu().numpy()
        # Gets the coordinates of all the foreground pixels
        foreground_coords = np.array(np.nonzero(foreground_mask))

        # Get the minimum x and the minimum y to generate the top left corner of the foreground
        top_left = np.min(foreground_coords, axis=1)
        # Get the maximum x and the maximum y to generate the bottom right corner of the foreground
        bottom_right = np.max(foreground_coords, axis=1)

        # Crops the image to be confined between the top left and bottom right coordinates we found
        # above
        # self.__3d_view = self.__3d_view[
        #     top_left[0] : bottom_right[0], top_left[1] : bottom_right[1], top_left[2] : bottom_right[2]
        # ]
        shape = self.__3d_view.shape
        self.__3d_view = (
            self.__3d_view[top_left[0] :, :, :]
            if bottom_right[0] + 1 == shape[0]
            else self.__3d_view[top_left[0] : bottom_right[0] + 1, :, :]
        )
        self.__3d_view = (
            self.__3d_view[:, top_left[1] :, :]
            if bottom_right[1] + 1 == shape[1]
            else self.__3d_view[:, top_left[1] : bottom_right[1] + 1, :]
        )
        self.__3d_view = (
            self.__3d_view[:, :, top_left[2] :]
            if bottom_right[2] + 1 == shape[2]
            else self.__3d_view[:, :, top_left[2] : bottom_right[2] + 1]
        )

    def contour_crop(self):

        rect_list = []

        original_shape = self.get_size()

        for ct_slice in self.get_images(plane=Plane.AXIAL):
            ct_slice = ct_slice.cpu().numpy().astype(np.float32).round()

            # Converts the image into a black and white image
            # The background is completely black and the foreground is completely white
            _, thresh = cv2.threshold(
                ct_slice, np.min(ct_slice), 255, cv2.THRESH_BINARY
            )
            # Convert all values in the thresholded image to unsigned 8 bit integers
            thresh = thresh.astype(np.uint8)

            # Find the contours in the thresholded image
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            if len(contours):
                contour_area = max(
                    [contour for contour in contours],
                    key=lambda contour: cv2.contourArea(contour),
                )

                x, y, w, h = cv2.boundingRect(contour_area)

                rect_list.append((x, y, w, h))
            else:
                rect_list.append((original_shape[1] // 2, original_shape[2] // 2, 0, 0))

            # fig, ax = plt.subplots()
            # ax.imshow(ct_slice, cmap="gray")
            # rect = Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
            # ax.add_patch(rect)

            # plt.show()

        bottom_left_y, bottom_left_x = min([rect[0] for rect in rect_list]), min(
            [rect[1] for rect in rect_list]
        )
        top_right_y, top_right_x = max([rect[0] + rect[2] for rect in rect_list]), max(
            [rect[1] + rect[3] for rect in rect_list]
        )

        shape = self.__3d_view.shape
        self.__3d_view = (
            self.__3d_view[:, bottom_left_x:, :]
            if top_right_x + 1 == shape[1]
            else self.__3d_view[:, bottom_left_x : top_right_x + 1, :]
        )
        self.__3d_view = (
            self.__3d_view[:, :, bottom_left_y:]
            if top_right_y + 1 == shape[1]
            else self.__3d_view[:, :, bottom_left_y : top_right_y + 1]
        )

        return original_shape

    def zoom_resize(self, desired_size):
        w, h = self.__3d_view.shape[1:]

        zoom_factor = min((desired_size[0] / w, desired_size[1] / h))
        zoom = Zoom(zoom=zoom_factor, keep_size=False)

        shape = self.__3d_view.shape
        zoom_img = torch.zeros(
            (shape[0], int(shape[1] * zoom_factor), int(shape[2] * zoom_factor))
        )
        for i in range(self.__3d_view.shape[0]):
            zoom_img[i, :, :] = zoom(self.__3d_view[i, :, :].unsqueeze(0))[0, :, :]

        self.pad_image(desired_size)

    def pad_image(self, desired_size):
        shape = self.__3d_view.shape

        padding_top = math.ceil((desired_size[0] - shape[1]) / 2)
        padding_bottom = math.floor((desired_size[0] - shape[1]) / 2)

        padding_left = math.ceil((desired_size[1] - shape[2]) / 2)
        padding_right = math.floor((desired_size[1] - shape[2]) / 2)

        self.__3d_view = torch.nn.functional.pad(
            self.__3d_view,
            (padding_left, padding_right, padding_top, padding_bottom, 0, 0),
            mode="constant",
            value=self.__3d_view.min().item(),
        )

    def rotate_image(self, plane=Plane.AXIAL):
        # if plane is Plane.AXIAL:
        #     return [torch.tensor(self.__3d_recon[i, :, :], requires_grad=require_grad) for i in range(self.__3d_recon.size(dim=0))]
        # elif plane is Plane.CORONAL:
        #     return [torch.tensor(self.__3d_recon[i, :, :], requires_grad=require_grad) for i in range(self.__3d_recon.size(dim=2))]
        # elif plane is Plane.SAGITTAL:
        #     return [torch.tensor(self.__3d_recon[i, :, :], requires_grad=require_grad) for i in range(self.__3d_recon.size(dim=1))]
        # else:
        #     raise ValueError("The argument plane is invalid")
        raise NotImplementedError

    def correct_image_tilt(self):
        raise NotImplementedError

    # https://stackoverflow.com/a/47736465
    # https://nanonets.com/blog/optical-flow/
    def interpolate(self, desired_depth, insert_extra_bottom=True):
        # TODO: Need to make it so it is less related to the previous slice

        # Prevent interpolation multiple times
        if self.__interpolated:
            raise Exception("You can only interpolate a CT scan once")

        # Get the slices
        slices = self.get_images(plane=Plane.AXIAL, require_grad=False)
        # Creates a list for the all the slice (with the interpolated ones) to be stored
        interp_slices = [slices[0].cpu().numpy()]

        # Creates an list with the number of slices to add between each pair of slices
        frames_to_add = [(desired_depth - len(slices)) // (len(slices) - 1)] * (
            len(slices) - 1
        )
        # Add the extra slices required to meet the desired depth to the bottom or top (based on
        # argument passed)
        for i in range((desired_depth - len(slices)) % (len(slices) - 1)):
            frames_to_add[-i if insert_extra_bottom else i] += 1

        # Create a numpy array of all the possible x and y coordinate values
        y, x = np.mgrid[0 : slices[0].shape[0], 0 : slices[0].shape[1]]
        # Combine the x and y numpy array to form a matrix with all the possible coordinates in
        # the image
        coords = np.dstack([x, y]).astype(np.float32)

        # For each pair of consecutive slices in the ct scan
        for i in range(len(slices) - 1):
            # Get the pair of slices
            prev_slice = slices[i].cpu().numpy().astype(np.float32)
            next_slice = slices[i + 1].cpu().numpy().astype(np.float32)

            # Use the farneback method to estimate the optical flow
            flow = cv2.calcOpticalFlowFarneback(
                next_slice,
                prev_slice,
                None,
                0.5,
                10,
                60,
                3,
                7,
                1.5,
                cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )

            # For each slice that need to be added
            for j in range(1, frames_to_add[i] + 1):
                # Generated the interpolated pixel map, moving the pixels the required portion of
                # the calculated optical flow
                interp_pixel_map = coords + (j / (frames_to_add[i] + 1)) * flow

                # Generated the interpolated slice
                interp_slice = cv2.remap(
                    prev_slice, interp_pixel_map, None, cv2.INTER_LINEAR
                )
                # Appends the next slice to the list of all the slices
                interp_slices.append(interp_slice)

            # Add the upper slice of the pair of slices to the list fo all the slices
            interp_slices.append(next_slice)

            self.__interpolated_blocks.append(interp_slices[-frames_to_add[i] - 2 :])

        # Updates the 3d view
        self.__3d_view = torch.tensor(np.array(interp_slices))

        self.__interpolated = True

    def remove_slices(self, num_to_remove):
        # 1. Divide into block
        # 2. Find random location
        # 3. Remove at that location and then insert a length of a block away
        # raise NotImplementedError

        num_slices = self.__3d_view.shape[0]

        #  = torch.Tensor((num_slices - num_to_remove, *self.__3d_view.shape[1:]))

        slices_to_keep = random.choices(
            [i for i in range(num_slices)], k=num_slices - num_to_remove
        )

        self.__3d_view = self.__3d_view[np.r_[slices_to_keep], :, :]

    def change_depth(self, desired_depth):
        if desired_depth > self.__3d_view.shape[0]:
            self.interpolate(desired_depth)
        elif desired_depth < self.__3d_view.shape[0]:
            self.remove_slices(self.__3d_view.shape[0] - desired_depth)

    def transform_and_save(self, runtime_id):
        def pad_image(image, desired_size):
            shape = image.shape

            padding_top = math.ceil((desired_size[0] - shape[1]) / 2)
            padding_bottom = math.floor((desired_size[0] - shape[1]) / 2)

            padding_left = math.ceil((desired_size[1] - shape[2]) / 2)
            padding_right = math.floor((desired_size[1] - shape[2]) / 2)

            return torch.nn.functional.pad(
                image,
                (padding_left, padding_right, padding_top, padding_bottom, 0, 0),
                mode="constant",
                value=image.min().item(),
            )

        def compose(transforms):
            return lambda tensor_list: [
                reduce(
                    lambda res, f: f(
                        res
                        if tensor.shape == res.shape[1:]
                        else pad_image(res, tensor.shape)
                    ),
                    transforms,
                    tensor.unsqueeze(0),
                )
                for tensor in tensor_list
            ]

        def save(tensor, path):
            dir_path = Path(path).parent
            dir_path.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)

        for i, ct_slice in enumerate(self.get_images()):
            save(
                ct_slice,
                f".\\output-data\\{self.__patient_name}\\scan-{runtime_id}\\augmented-data-0-0\\slice-{i}.tensor",
            )

        for i in range(1, len(TRANSFORMATIONS) + 1):
            for j, comb in enumerate(combinations(TRANSFORMATIONS, r=i)):
                transforms_func = compose(comb)
                # augmented_ct_scans.append(transforms_func(self.get_images()))
                for k, ct_slice in enumerate(transforms_func(self.get_images())):
                    save(
                        ct_slice,
                        f".\\output-data\\{self.__patient_name}\\scan-{runtime_id}\\augmented-data-{i}-{j}\\slice-{k}.tensor",
                    )
                    

    def get_interpolated_blocks(self):
        return self.__interpolated_blocks

    def convert_to_nrrd(self, output_file):
        def convert(command_args, verbose = True, path_to_log_file = None, return_bash_command = False):
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
                bash_command += ["--%s"%(key), val]
            
            if verbose:
                print("\nRunning 'plastimatch convert' with the specified arguments:")
                for key, val in command_args.items():
                    print("  --%s"%(key), val)
            
            try:
                if path_to_log_file:
                    with open(path_to_log_file, "a") as log_file:
                        bash_exit_status = subprocess.run(bash_command,
                                                        stdout = log_file, stderr = log_file,
                                                        check = True, shell = True)
                else:
                    # if no log file is specified, output to the default stdout and stderr
                    bash_exit_status = subprocess.run(bash_command, capture_output = True, check = True, shell = True)
                    print(bash_exit_status.stdout)
                    print(bash_exit_status.stderr)
                    
                    if verbose: print("... Done.")
                
            except Exception as e:
                # if the process exits with a non-zero exit code, a CalledProcessError exception will be raised
                # attributes of that exception hold the arguments, the exit code, and stdout and stderr if they were captured
                # For details, see: https://docs.python.org/3/library/subprocess.html#subprocess.run
                
                # FIXME: return exception?
                print(e)
            
            if return_bash_command:
                return bash_command
        
        
        convert(command_args={
                "input": str(os.path.abspath(self.get_path())),
                # "input": r"C:\Users\Nisha\OneDrive\Documents\GitHub\osteoporosis_ct\notebooks\DeepContrast\data",
                "output-img": str(os.path.abspath(output_file)),
                "algorithm": "itk"
            }
        )
        
        # nrrd.write(output_file, data)
        

if __name__ == "__main__":
    start = time.time()
    
    ctscan = CTScan.read_data(r".\notebooks\DeepContrast\data")[0]
    print("converting")
    ctscan.convert_to_nrrd(r"C:\Users\Nisha\OneDrive\Documents\GitHub\osteoporosis_ct\notebooks\DeepContrast\test.nrrd")

    proj_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(proj_dir, 'models')

    df_img, img_arr = data_prepro(
        body_part="HeadNeck",
        data_dir=r"C:\Users\Nisha\OneDrive\Documents\GitHub\osteoporosis_ct\notebooks\DeepContrast",
    )
    
    model_pred(
        body_part="HeadNeck",
        save_csv=True,
        model_dir=model_dir,
        out_dir=proj_dir,
        df_img=df_img,
        img_arr=img_arr
        )
    
    print(f"Run time: {time.time() - start}")