"""
pyMFI.py

Description:
    The goal of this script is to calculate mean fluorescence intensity value (MFI) for each cell segmented by the 10X Xenium in-situ gene expression analysis. 

    This script takes in two channel immunofluorecence images (ome.tiff), 10X xenium segmentations (cells.zarr.zip), and a transformation matrix (.csv) used to align the IF and Xenium morphology image
     
    Output is a '.csv' file containing cell ids, mean, min, and max intensity values for each cell in the experiment

Usage:

pyMFI_v3.py <IFimage.ome.tif>  \
<path/to/cells.zarr.zip> \
<path/to/alignment_matrix.csv> \
--output_file_path path/to/ouput.csv \
--segment_type "cells" \
--pix_size 0.2125 \
--num_workers 8

Author:
jrose

Date modified:
19Mar2024

"""

import zarr
import os
import csv 
import numpy as np
import argparse
from skimage import io
from skimage.transform import AffineTransform
from skimage.draw import polygon
import tifffile
from multiprocessing import Pool, cpu_count, shared_memory


########## File input from zarr

def open_zarr(path: str) -> zarr.Group:
    """Opens a Zarr file from a zip or directory store."""
    store = (
        zarr.ZipStore(path, mode="r") 
        if path.endswith(".zip") 
        else zarr.DirectoryStore(path)
    )
    return zarr.group(store=store)


########## Load image to shared memory 

def load_image_shared(image_path, channel=1):
    # Load the image and select the specified channel
    image = tifffile.imread(image_path)[channel]
    
    # Create shared memory for the image
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
    
    # Copy the image data into the shared memory
    shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
    np.copyto(shared_image, image)
    
    return shm, image.shape, image.dtype

########## Converting cell id prefix to string
"""The cell id from 10X Xenium outputs needs to be converted from numeric to string using the following"""

def shift_hex_characters(hex_string):
    # Create a translation table to map hex digits to a-p range
    translation_table = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    # Translate the hex string using the table
    return hex_string.translate(translation_table)

def convert_to_string(cell_id_prefix, dataset_suffix):
     # Convert cell_id_prefix to hexadecimal, pad to 8 characters with leading zeros, and remove '0x'
    hex_rep = f"{cell_id_prefix:08x}"
    # Shift hex characters to the new range
    shifted_hex = shift_hex_characters(hex_rep)
    # Concatenate shifted_hex and dataset_suffix with a dash
    result = f"{shifted_hex}-{dataset_suffix}"
    return result

def process_id_array(cell_id):
    # Initialize an empty list to store the output strings
    output_strings = []

    # Loop through each row in the array
    for row in cell_id:
        cell_id_prefix = row[0]
        dataset_suffix = row[1]
        # Convert each row to the desired string format
        output_string = convert_to_string(cell_id_prefix, dataset_suffix)
        # Append the result to the list
        output_strings.append(output_string)

    # Convert the list of strings to a numpy array
    return np.array(output_strings)

########## Load polygon data

def load_polygon_data(zarr_file, segment_type, pix_size):
    """Loads polygon data (cell_index, vertices) from a given group (0 for nuclei, 1 for cell boundaries)."""
    if segment_type == "cells":
        zarr_group = "1"
    elif segment_type == "nuclei":
        zarr_group = "0"
    else:
        raise ValueError("Error: Please use either 'cells' or 'nuclei' for segment_type parameter")

    group = zarr_file[f'polygon_sets/{zarr_group}']
    
    cell_id = zarr_file['cell_id']
    cell_indices = process_id_array(cell_id)
    
    num_vertices = group['num_vertices'][:]
    vertices = group['vertices'][:] / pix_size
    #^Use pixel size toconvert from physical units to pxiels
    
    return cell_indices, num_vertices, vertices

########## Transform polygons to new image coordinates

def apply_affine_transformation(vertices, num_vertices, alignment_matrix, inverse=True):
    """Transforms polygon coordinates from Xenium morphology image to new IF image coordinates"""
    transform = AffineTransform(matrix=alignment_matrix)
    if inverse == True:
        transform = transform.inverse

    transformed_polygons = []
    
    for poly_idx, nv in enumerate(num_vertices):
        # Extract valid (x, y) pairs for the current polygon up to nv vertices
        polygon = vertices[poly_idx, :nv*2].reshape((nv, 2))  # Shape (nv, 2)
        
        # Apply the affine transformation to the polygon vertices
        transformed_polygon = transform(polygon)  
        
        # Store only the transformed (x, y) coordinates
        transformed_polygons.append(transformed_polygon[:, :2])  # Shape (nv, 2)
    
    return transformed_polygons

########## Calculates MFI

def calculate_MFI_stats(image, polygon_coords):
    """Calculates the statistics around pixel intensity values within polygon boundaries"""
    #image_chl2 = image[channel] #For my images channel index 1 is Cy5, index 0 is usually DAPI

    x_coords, y_coords = polygon_coords[:, 0], polygon_coords[:, 1]

    rr, cc = polygon(np.clip(y_coords, 0, image.shape[0] - 1), 
                     np.clip(x_coords, 0, image.shape[1] - 1))
    
    # Check if the polygon area contains valid indices
    if len(rr) == 0 or len(cc) == 0 or image[rr, cc].size == 0:
        # Return NaN values if no valid pixels are found within the polygon
        return float('nan'), float('nan'), float('nan')
    
    MFI = np.mean(image[rr, cc])
    min_int = np.min(image[rr, cc])
    max_int = np.max(image[rr, cc])

    return MFI, min_int, max_int

########## Parallel Processing

def process_polygon_chunk(shm_name, image_shape, image_dtype, polygons_chunk):
    # Access the shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create a numpy array backed by shared memory
    shared_image = np.ndarray(image_shape, dtype=image_dtype, buffer=shm.buf)
    
    intensity_stats = []
    for polygon in polygons_chunk:
        intensity_stat = calculate_MFI_stats(shared_image, polygon)
        intensity_stats.append(intensity_stat)
    return intensity_stats

def parallel_process_polygons(shm_name, image_shape, image_dtype, polygons, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    chunk_size = len(polygons) // num_workers
    #chunk_size=50000
    polygon_chunks = [polygons[i:i + chunk_size] for i in range(0, len(polygons), chunk_size)]
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_polygon_chunk, [(shm_name, image_shape, image_dtype, chunk) for chunk in polygon_chunks])
    return [item for sublist in results for item in sublist]


########## Main function

def main(image_path, zarr_path, alignment_matrix_path, output_file_path, segment_type='cells', pix_size=0.2125, num_workers=None, channel=1):
    if output_file_path is None:
        output_file_path = os.path.join(os.getcwd(), "pyMFI_output_results.csv")
    
    # Load image
    shm, image_shape, image_dtype = load_image_shared(image_path, channel=channel)
    
    # Load Zarr data
    zarr_file = open_zarr(zarr_path)
    cell_indices, num_vertices, vertices = load_polygon_data(zarr_file, segment_type, pix_size)
    
    # Load the alignment matrix
    alignment_matrix = np.loadtxt(alignment_matrix_path, delimiter=',')
    
    # Apply affine transformation
    transformed_polygons = apply_affine_transformation(vertices, num_vertices, alignment_matrix)
    
    # Calculate mean intensities in parallel
    intensity_stats = parallel_process_polygons(shm.name, image_shape, image_dtype, transformed_polygons, num_workers)
    
    # Write results to the output .csv file
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["cell_id", "MFI","Min Intensity", "Max Intensity"])  # Write the header row
        for cell_id, (mean_intensity, min_intensity, max_intensity) in zip(cell_indices, intensity_stats):
            writer.writerow([cell_id, mean_intensity, min_intensity, max_intensity])  # Write each row with cell ID and mean intensity
    
    print(f"Results saved to {output_file_path}")
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process images and Xenium cell segment polygons to compute MFI, min, and max intensities.")
    parser.add_argument("image_path", type=str, help="Path to the image file (.ome.tif)")
    parser.add_argument("zarr_path", type=str, help="Path to the cells.zarr.zip file containing 10X Xenium cell segmentation polygon data")
    parser.add_argument("alignment_matrix_csv", type=str, help="Path to the CSV file containing the 3x3 image alignment matrix")
    
    # Optional arguments
    parser.add_argument("--output_file_path", type=str, default=None, help="Path to save the output CSV file (default: current directory)")
    parser.add_argument("--segment_type", type=str, default='cells', help="Cell or nuclei segments in Xenium output (default: 'cells' for cell segments. Pass 'nuclei' for nuclei segments)")
    parser.add_argument("--pix_size", type=float, default=0.2125, help="Pixel size (um/pixel) for morphology image scaling (default: 0.2125)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers (default: 8. Pass None for max available - 1)")
    parser.add_argument("--channel", type=int, default=1, help="Channel index to use for MFI calculation (default: 1)")


    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(
        image_path=args.image_path,
        zarr_path=args.zarr_path,
        alignment_matrix_path=args.alignment_matrix_csv,
        output_file_path=args.output_file_path,
        segment_type=args.segment_type,
        pix_size=args.pix_size,
        num_workers=args.num_workers,
        channel=args.channel
    )

    # Example usage (for testing)
    #image_path = "/santangelo-lab/JimR/Xenium/XEN1_CRE_P81_Mouse_Lung/IF/Cre_P81_2_Lung_Xenslide2.ome.tif"
    #zarr_path = "/data-raid/Xenium/20241010__184320__XEN1___Cre_P81_Mouse_Lung_101024/output-XETG00215__0033995__P81-2-Lung__20241010__184343-2/cells.zarr.zip"
    #alignment_matrix_path = "/santangelo-lab/JimR/Xenium/XEN1_CRE_P81_Mouse_Lung/IF/Cre_P81_2_Lung_Xenslide2_matrix.csv"
    
    #main(image_path, zarr_path, alignment_matrix_path, output_file_path=None, segment_type='1', pix_size=0.2125, num_workers=8)

######################
