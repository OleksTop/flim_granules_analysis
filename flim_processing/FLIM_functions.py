def analyze_single_fitted_channel(intensity_image_path, fit_image_path,
                                  minimal_mean_intensity=300, minimal_intensity=100,
                                  minimal_chi2=89, maximal_chi2=120):
    """
    Analyzes a single channel of lifetime fitted data from microscopy images.

    This function performs analysis on a single channel of fitted data from microscopy images,
    corresponding to imaging of a single fluorophore at Leica SP8 software. The function assumes
    that the fitted data represents lifetime in picoseconds (original data scaled by 1000).

    Important note: The chi-square value provided is scaled by 100.

    Args:
        intensity_image_path (str): Path to the intensity image file.
        fit_image_path (str): Path to the fit image file.
        minimal_mean_intensity (int, optional): Minimum mean intensity threshold for object filtering. Defaults to 300.
        minimal_intensity (int, optional): Minimum intensity threshold for object filtering. Defaults to 100.
        minimal_chi2 (float, optional): Minimum chi-square value threshold for fitting quality. Defaults to 89.
        maximal_chi2 (float, optional): Maximum chi-square value threshold for fitting quality. Defaults to 120.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: Filtered data table with properties of identified objects.
            - pandas.DataFrame: Report table summarizing analysis results.
            - numpy.ndarray: Array of fitted lifetimes in picoseconds corresponding to each object.

    Raises:
        ValueError: If the intensity or fit image paths are invalid or cannot be loaded.

    Notes:
        - The function assumes that intensity and fit images are correctly aligned and processed.
        - Lifetime values are assumed to be in picoseconds based on the fitting model used.
        - The chi-square value is scaled by 100 to facilitate to avoid float.

    """

    from aicsimageio import AICSImage  # to reassign order of dimensions
    from skimage.io import imread
    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import numpy as np
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from skimage.measure import regionprops_table
    import pandas as pd
    from pathlib import Path
    from napari_skimage_regionprops._parametric_images import relabel_with_map_array
    from FLIM_functions import segment_image

    def standard_deviation_intensity(region, intensities):
        return np.std(intensities[region])

    # Open the data files (this is implementation for 1 channel image)
    aics_image1 = AICSImage(intensity_image_path)
    aics_image2 = AICSImage(fit_image_path)

    counts_cfp = aics_image1.get_image_data("XY", C=0)
    chi_cfp = aics_image2.get_image_data("XY", C=0)
    tau_cfp = aics_image2.get_image_data("XY", C=1)

    image1_M, image4_elosrv1 = segment_image(counts_cfp)

    multichannel_image = np.stack([image1_M, chi_cfp, tau_cfp], axis=-1)

    # Get measurements from objects
    properties = regionprops_table(image4_elosrv1, multichannel_image,
                                   properties=['label', 'area', 'intensity_mean', 'intensity_min'],
                                   extra_properties=[standard_deviation_intensity])

    df = pd.DataFrame(properties)
    df.rename(columns={'label': 'Label', 'area': 'Area',
                       'intensity_mean-0': 'intensity_mean_fluor',
                       'intensity_mean-1': 'mean_chi2',
                       'intensity_mean-2': 'mean_tau',
                       'standard_deviation_intensity-0': 'STD_fluor',
                       'standard_deviation_intensity-1': 'STD_chi2',
                       'standard_deviation_intensity-2': 'STD_tau',
                       'intensity_min-0': 'min_fluor',
                       'intensity_min-1': 'min_chi2',
                       'intensity_min-2': 'min_tau'}, inplace=True)

    # Filtering the labels according to their chi value and intensity min and mean
    table_mask = (df['intensity_mean_fluor'] > minimal_mean_intensity) & \
                 (df['mean_chi2'] < maximal_chi2) & \
                 (df['mean_chi2'] > minimal_chi2) & \
                 (df['min_fluor'] > minimal_intensity)
    result_filtered = df[table_mask]

    # Create an image with intensity, fitted lifetimes and objects that are re-labelled according to their mean lifetimes
    param_image = relabel_with_map_array(image4_elosrv1, result_filtered['Label'], result_filtered['mean_tau'])
    report_multichannel_image = np.stack([counts_cfp, tau_cfp, param_image], axis=-1)

    # Optional brief estimation on how was the quality of the data
    data_filtered = df.shape[0] - result_filtered.shape[0]
    proc_filtered = (result_filtered.shape[0] * 100) / df.shape[0]

    # Build reports table
    report_table = pd.DataFrame()
    report_table['Mean_fluor_intensity'] = result_filtered['intensity_mean_fluor'].mean(),
    report_table['Mean_tau'] = result_filtered['mean_tau'].mean(),
    report_table['Mean_STD'] = result_filtered['STD_tau'].std(),
    report_table['Number_of_objects'] = result_filtered.shape[0],
    report_table['Initial_number_of_objects'] = df.shape[0]
    report_table['%_of_object_remained'] = proc_filtered,

    return result_filtered, report_table, report_multichannel_image


def segment_image(intensity_image, median_filter_size=2, top_hat_radius=5, minimal_object_size=30):
    """
    Segment objects in an intensity image for further analysis.

    This function processes an intensity image to segment objects based on specified parameters.

    Args:
        intensity_image (numpy.ndarray): Intensity image to be processed.
        median_filter_size (int, optional): Size of the median filter for noise reduction. Defaults to 2.
        top_hat_radius (int, optional): Radius of the top-hat filter for background correction. Defaults to 5.
        minimal_object_size (int, optional): Minimum object size for segmentation. Defaults to 30.

    Returns:
        tuple: Tuple containing:
            - numpy.ndarray: Processed image after segmentation and filtering.
            - numpy.ndarray: Binary mask of segmented objects.

    Notes:
        - Uses various image processing techniques from different libraries.
        - Assumes input image is suitable for segmentation.
    """

    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    import numpy as np

    # Intensity image processing
    image1_M = nsitk.median_filter(intensity_image, median_filter_size, median_filter_size, 0)
    image2_ths = cle.top_hat_sphere(image1_M, None, top_hat_radius, top_hat_radius, 0)

    binary_mask = np.asarray(cle.threshold_otsu(image2_ths))

    # Post-processing steps
    image1_S = nsbatwm.split_touching_objects(binary_mask, 2.0)
    objects = nsitk.touching_objects_labeling(binary_mask)

    # Exclude labels outside size range
    image4_elosrv1 = np.asarray(cle.exclude_small_labels(objects, None, minimal_object_size))

    return image1_M, image4_elosrv1




def generate_small_report_fig(report_multichannel_image, image_id, output_folder_path=None):
    """
    Generates a small report figure from a multichannel image.
    
    Args:
        report_multichannel_image (numpy.ndarray): Multichannel image to generate the report from.
        image_id (str): Identifier used for describing and naming the figure.
        output_folder_path (str or Path, optional): Path to the output folder where the image will be saved.
            If None, the image will be saved locally. Defaults to None.

    Returns:
        None
    """
        
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Define default output directory if output_folder_path is None
    if output_folder_path is None:
        output_folder_path = Path(".")

    fig, axes = plt.subplots(1, 3, squeeze=False, figsize=(15, 8))
    
    for i in range(3):
        axes[0, i].imshow(report_multichannel_image[:, :, i])
        axes[0, i].axis("off")
    
    fig.suptitle(str(image_id))
    plt.tight_layout()

    # Construct the filename
    filename = f"{image_id}_plots.png"

    # Save the figure in the specified output folder or locally
    fig.savefig(output_folder_path / filename)
    plt.close()

    
def generate_small_report_fig_coloc(report_multichannel_image, image_id, output_folder_path=None):
    """
    Generates a small report figure from a multichannel image (with colocalization analysis of 2 chanells).

    Args:
        report_multichannel_image (numpy.ndarray): Multichannel image to generate the report from.
        image_id (str): Identifier used for describing and naming the figure.
        output_folder_path (str or Path, optional): Path to the output folder where the image will be saved.
            If None, the image will be saved locally. Defaults to None.

    Returns:
        None
    """
    
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Define default output directory if output_folder_path is None
    if output_folder_path is None:
        output_folder_path = Path(".")

    fig, axes = plt.subplots(1, 4, squeeze=False, figsize=(15, 8))
    
    for i in range(4):
        axes[0, i].imshow(report_multichannel_image[:, :, i])
        axes[0, i].axis("off")
    
    fig.suptitle(str(image_id))
    plt.tight_layout()

    # Construct the filename
    filename = f"{image_id}_plots.png"

    # Save the figure in the specified output folder or locally
    fig.savefig(output_folder_path / filename)
    plt.close()





def analyze_single_fastflim_channel(intensity_image_path, minimal_mean_intensity=300, minimal_intensity=100):
    """
    Analyzes FAST flim image from a single channel microscopy image.
    The order of channels corresponds to Leica SP8 software, and lifetimes are multiplied by 100, 
    so time is represented in picoseconds.

    Args:
        intensity_image_path (str): Path to the intensity image file.
        minimal_mean_intensity (int, optional): Minimum mean intensity threshold. Defaults to 300.
        minimal_intensity (int, optional): Minimum intensity threshold. Defaults to 100.

    Returns:
        tuple: Tuple containing:
            - pandas.DataFrame: Filtered data table with properties of objects.
            - pandas.DataFrame: Report table summarizing analysis results.
            - numpy.ndarray: Multichannel image with intensity, fitted lifetimes, and labeled objects.
    """
    
    from aicsimageio import AICSImage #to reassign order of dimensions
    from skimage.io import imread
    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import numpy as np
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from skimage.measure import regionprops_table
    import pandas as pd
    from pathlib import Path
    from napari_skimage_regionprops._parametric_images import relabel_with_map_array
    def standard_deviation_intensity(region, intensities):
             return np.std(intensities[region])
    from FLIM_functions import segment_image
        
    #open the data files (this is implementatio for 1 chanell image)
    aics_image1 = AICSImage(intensity_image_path)

    counts_cfp = aics_image1.get_image_data("XY", C=0) 
    tau_cfp = aics_image1.get_image_data("XY", C=1)

    image1_M,image4_elosrv1=segment_image(counts_cfp)

    multichannel_image = np.stack([image1_M, tau_cfp], axis=-1)

    #get measurements from objects 
    properties = regionprops_table(image4_elosrv1, multichannel_image, 
                                           properties = ['label', 'area','intensity_mean','intensity_min'],
                                           extra_properties=[standard_deviation_intensity])

    df = pd.DataFrame(properties)
    df.rename(columns={'label': 'Label','area': 'Area',
                              'intensity_mean-0': 'intensity_mean_fluor',
                              'intensity_mean-1': 'mean_tau',
                              'standard_deviation_intensity-0': 'STD_fluor',
                              'standard_deviation_intensity-1': 'STD_tau',
                              'intensity_min-0': 'min_fluor',
                              'intensity_min-1': 'min_tau'},inplace=True)

    #filtering the labels according to their chi value and intensity min ans mean

    table_mask = (df['intensity_mean_fluor'] > minimal_mean_intensity) & (df['min_fluor'] > minimal_intensity) #& (df_counts['Area'] > 30)
    result_filtered = df[table_mask]

    #Optional brief estimation on how was the quality of the data 
    #data_filtered=(df.shape[0])-(result_filtered.shape[0]) 
    proc_filtered=(((result_filtered.shape[0])*100)/(df.shape[0]))
    
    # Buidl reposrts table 
    report_table = pd.DataFrame()
    report_table['Mean_fluor_intensity']=result_filtered['intensity_mean_fluor'].mean(),
    report_table['Mean_tau'] = result_filtered['mean_tau'].mean(),
    report_table['Mean_STD'] = result_filtered['STD_tau'].std(),
    report_table['Number_of_objects'] = result_filtered.shape[0],
    report_table['Initial_number_of_objects'] = df.shape[0]
    report_table['%_of_object_remained'] = proc_filtered,
    report_table['File Name'] = intensity_image_path.stem
    
    #Create an output image, intensity channel, lifetime fitted data (to additionaly check if chanells are correct)
    #and objects that are re-labelled according to teir mean lifetimes 
    
    #is it possible to save already with the correct cmap?
    
    param_image=relabel_with_map_array(image4_elosrv1,result_filtered['Label'],result_filtered['mean_tau'])
    report_multichannel_image = np.stack([counts_cfp, tau_cfp, param_image], axis=-1)
    
    return result_filtered, report_table, report_multichannel_image



def analyze_fitted_coloc(intensity_image_path, fit_image_path, minimal_mean_intensity=300, minimal_intensity=100, minimal_chi2=89, maximal_chi2=120):
    """
    Analyzes fitted FLIM data for a specified fluorophore
    The order of channels corresponds to Leica SP8 software, and lifetimes are multiplied by 100, 
    so time is represented in picoseconds.

    Args:
        intensity_image_path (str): Path to the intensity image file.
        fit_image_path (str): Path to the fit image file.
        minimal_mean_intensity (int, optional): Minimum mean intensity threshold. Defaults to 300.
        minimal_intensity (int, optional): Minimum intensity threshold. Defaults to 100.
        minimal_chi2 (float, optional): Minimal chi-square value threshold. Defaults to 89.
        maximal_chi2 (float, optional): Maximal chi-square value threshold. Defaults to 120.

    Returns:
        tuple: Tuple containing:
            - pandas.DataFrame: Filtered data table with properties of objects.
            - pandas.DataFrame: Report table summarizing analysis results.
            - numpy.ndarray: Multichannel image with intensity, fitted lifetimes, and labeled objects.
    """
        
    import napari
    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import matplotlib.pyplot as plt
    import numpy as np
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from skimage.measure import label, regionprops_table
    from aicsimageio import AICSImage #to reassign order of dimensions
    import matplotlib.pyplot as plt
    from skimage.io import imshow
    from skimage.io import imread
    import numpy as np
    import pandas as pd
    from skimage.measure import label, regionprops_table
    def standard_deviation_intensity(region, intensities):
         return np.std(intensities[region])
    from napari_skimage_regionprops._parametric_images import relabel_with_map_array
    
    from FLIM_functions import segment_image
    
    
    # Load images using AICSImage
    aics_image1 = AICSImage(intensity_image_path)
    aics_image2 = AICSImage(fit_image_path)

    # Extract image data for counts and chi values
    counts_cfp = aics_image1.get_image_data("XY", C=0) # Values without manual threshold for chi & tau
    counts_sir = aics_image1.get_image_data("XY", C=2)

    tau_cfp = aics_image2.get_image_data("XY", C=1) # Use data from aics_image2
    chi_cfp = aics_image2.get_image_data("XY", C=0)    # Values after manual threshold on 'counts_cfp'


    image1_M,image4_elosrv1=segment_image(counts_sir)

    #the important difference between 1 chanell and 2!
    #preparation of the second  image, that is used to 'derive' data from  

    median_counts_cfp = nsitk.median_filter(counts_cfp, 1, 1, 0) #but bin of 1 on sp8
    thb_counts_cfp = np.asarray(cle.threshold_otsu(median_counts_cfp)) 

    #create a mask with intersection of  objects from the segmentation with the second image 
    image2_ba = cle.binary_and(thb_counts_cfp,image4_elosrv1)
    objects_labels3 = label(image2_ba) # labeles intercection mask 
    multichannel_image = np.stack([median_counts_cfp, chi_cfp, tau_cfp], axis=-1)

    #get measurements from objects 
    properties = regionprops_table(image4_elosrv1, multichannel_image, 
                                           properties = ['label', 'area','intensity_mean','intensity_min'],
                                           extra_properties=[standard_deviation_intensity])

    df = pd.DataFrame(properties)
    df.rename(columns={'label': 'Label','area': 'Area',
                              'intensity_mean-0': 'intensity_mean_fluor',
                              'intensity_mean-1': 'mean_chi2',
                              'intensity_mean-2': 'mean_tau',
                              'standard_deviation_intensity-0': 'STD_fluor',
                              'standard_deviation_intensity-1': 'STD_chi2',
                              'standard_deviation_intensity-2': 'STD_tau',
                              'intensity_min-0': 'min_fluor',
                              'intensity_min-1': 'min_chi2',
                              'intensity_min-2': 'min_tau'},inplace=True)

    #filtering the labels according to their chi value and intensity min ans mean

    table_mask = (df['intensity_mean_fluor'] > minimal_mean_intensity) & (df['mean_chi2'] < maximal_chi2)  & (df['mean_chi2'] > minimal_chi2) & (df['min_fluor'] > minimal_intensity) #& (df_counts['Area'] > 30)
    result_filtered = df[table_mask]
    
    #Create an image with intensity, fitted lifetimes and objects that are re-labelled according to teir mean lifetimes 
    #is it possible to save already with the correct cmap?
    
    param_image=relabel_with_map_array(image4_elosrv1,result_filtered['Label'],result_filtered['mean_tau'])
    report_multichannel_image = np.stack([counts_cfp, tau_cfp, image1_M, param_image], axis=-1)
    
    #Optional brief estimation on how was the quality of the data 
    data_filtered=(df.shape[0])-(result_filtered.shape[0]) 
    proc_filtered=(((result_filtered.shape[0])*100)/(df.shape[0]))
    
    # Buidl reposrts table 
    report_table = pd.DataFrame()
    report_table['Mean_fluor_intensity']=result_filtered['intensity_mean_fluor'].mean(),
    report_table['Mean_tau'] = result_filtered['mean_tau'].mean(),
    report_table['Mean_STD'] = result_filtered['STD_tau'].std(),
    report_table['Number_of_objects'] = result_filtered.shape[0],
    report_table['Initial_number_of_objects'] = df.shape[0]
    report_table['%_of_object_remained'] = proc_filtered,
    #report_table['File Name'] = intensity_image_path.stem
    
    

    return result_filtered, report_table, report_multichannel_image


def analyze_coloc_fastflim(intensity_image_path, minimal_mean_intensity=300, minimal_intensity=100):
    """
    Analyzes fast FLIM data for a specified fluorophore from colocalizing segmented objects, based on their fluorescence intensity images.
    The order of channels corresponds to Leica SP8 software, and lifetimes are multiplied by 100, 
    so time is represented in picoseconds.

    Args:
        intensity_image_path (str): Path to the intensity image file.
        fit_image_path (str): Path to the fit image file.
        minimal_mean_intensity (int, optional): Minimum mean intensity threshold. Defaults to 300.
        minimal_intensity (int, optional): Minimum intensity threshold. Defaults to 100.
        minimal_chi2 (float, optional): Minimal chi-square value threshold. Defaults to 89.
        maximal_chi2 (float, optional): Maximal chi-square value threshold. Defaults to 120.

    Returns:
        tuple: Tuple containing:
            - pandas.DataFrame: Filtered data table with properties of objects.
            - pandas.DataFrame: Report table summarizing analysis results.
            - numpy.ndarray: Multichannel image with intensity, fitted lifetimes, and labeled objects.
    """
    import napari
    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import matplotlib.pyplot as plt
    import numpy as np
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from skimage.measure import label, regionprops_table
    from aicsimageio import AICSImage #to reassign order of dimensions
    import matplotlib.pyplot as plt
    from skimage.io import imshow
    from skimage.io import imread
    import numpy as np
    import pandas as pd
    from skimage.measure import label, regionprops_table
    def standard_deviation_intensity(region, intensities):
         return np.std(intensities[region])
    from napari_skimage_regionprops._parametric_images import relabel_with_map_array

    from FLIM_functions import segment_image
        
    #open the data files (this is implementatio for 1 chanell image)
    aics_image1 = AICSImage(intensity_image_path)

    # Extract image data for counts and chi values
    counts_cfp = aics_image1.get_image_data("XY", C=0) # Values without manual threshold for chi & tau
    counts_sir = aics_image1.get_image_data("XY", C=2)
    tau_cfp = aics_image1.get_image_data("XY", C=1) # Use data from aics_image1
    
    image1_M,image4_elosrv1=segment_image(counts_sir)

    #the important difference between 1 chanell and 2!
    #preparation of the second  image, that is used to 'derive' data from  

    median_counts_cfp = nsitk.median_filter(counts_cfp, 1, 1, 0) #but bin of 1 on sp8
    thb_counts_cfp = np.asarray(cle.threshold_otsu(median_counts_cfp)) 

    #create a mask with intersection of  objects from the segmentation with the second image 
    image2_ba = cle.binary_and(thb_counts_cfp,image4_elosrv1)
    objects_labels3 = label(image2_ba) # labeles intercection mask 

    multichannel_image = np.stack([image1_M, tau_cfp], axis=-1)

    #get measurements from objects 
    properties = regionprops_table(image4_elosrv1, multichannel_image, 
                                           properties = ['label', 'area','intensity_mean','intensity_min'],
                                           extra_properties=[standard_deviation_intensity])

    df = pd.DataFrame(properties)
    df.rename(columns={'label': 'Label','area': 'Area',
                              'intensity_mean-0': 'intensity_mean_fluor',
                              'intensity_mean-1': 'mean_tau',
                              'standard_deviation_intensity-0': 'STD_fluor',
                              'standard_deviation_intensity-1': 'STD_tau',
                              'intensity_min-0': 'min_fluor',
                              'intensity_min-1': 'min_tau'},inplace=True)

    #filtering the labels according to their chi value and intensity min ans mean

    table_mask = (df['intensity_mean_fluor'] > minimal_mean_intensity) & (df['min_fluor'] > minimal_intensity) #& (df_counts['Area'] > 30)
    result_filtered = df[table_mask]

    #Optional brief estimation on how was the quality of the data 
    #data_filtered=(df.shape[0])-(result_filtered.shape[0]) 
    proc_filtered=(((result_filtered.shape[0])*100)/(df.shape[0]))
    
    # Buidl reposrts table 
    report_table = pd.DataFrame()
    report_table['Mean_fluor_intensity']=result_filtered['intensity_mean_fluor'].mean(),
    report_table['Mean_tau'] = result_filtered['mean_tau'].mean(),
    report_table['Mean_STD'] = result_filtered['STD_tau'].std(),
    report_table['Number_of_objects'] = result_filtered.shape[0],
    report_table['Initial_number_of_objects'] = df.shape[0]
    report_table['%_of_object_remained'] = proc_filtered,
    report_table['File Name'] = intensity_image_path.stem
    
    #Create an output image, intensity channel, lifetime fitted data (to additionaly check if chanells are correct)
    #and objects that are re-labelled according to teir mean lifetimes 
    
    #is it possible to save already with the correct cmap?
    
    param_image=relabel_with_map_array(image4_elosrv1,result_filtered['Label'],result_filtered['mean_tau'])
    report_multichannel_image = np.stack([counts_cfp, tau_cfp, image1_M, param_image], axis=-1)
    
    return result_filtered, report_table, report_multichannel_image

def analize_with_phasor(data_path_1,threshold = 50):
    """
    Analyzes FLIM data using phasor analysis from napri plugin napari_flim_phasor_plotter

    Args:
        data_path_1 (str): Path to the FLIM data file in ptu format 
        threshold (int, optional): Threshold value for background removal. Defaults to 50.

    Returns:
        tuple: Tuple containing:
            - pandas.DataFrame: Dataframe with features extracted from objects.
            - numpy.ndarray: Multichannel image with intensity, G, S, and phasor lifetime.
    """
        
    import napari_flim_phasor_plotter as flim_plot
    import numpy as np
    import napari
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from skimage.measure import label, regionprops_table
    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import pandas as pd
    from napari_skimage_regionprops._parametric_images import relabel_with_map_array
    from FLIM_functions import segment_image
    
    #Loading PTU files and reading the images with correct chanells 
    list_layerdatatuple_1 = flim_plot._reader.read_single_ptu_file(data_path_1)
    list_layerdatatuple_1[0].shape
    
    # Get arrays and metadata for image1
    image1_channel0 = list_layerdatatuple_1[0][0]
    image1_channel1 = list_layerdatatuple_1[0][1]
    image1_metadata0 = list_layerdatatuple_1[1][0]
    image1_metadata1 = list_layerdatatuple_1[1][1]

    # Conditional statements to select the needed channel
    if image1_channel0 is not None and np.any(image1_channel0): 
        image_1 = image1_channel0
        image_1_metadata = image1_metadata0
    else:
        image_1 = image1_channel1
        image_1_metadata = image1_metadata1

    # Get laset frequency from metadata
    laser_frequency = image_1_metadata['TTResult_SyncRate'] * 1E-6
    #print(f"Laser frequency: {laser_frequency} MHz")

    # Apply time mask (getting rid of the 'rising'part of the exponent from the entire image)
    time_mask = flim_plot.filters.make_time_mask(image_1, laser_frequency=laser_frequency)
    image_1_time_sliced = image_1[time_mask]

    #optional tresholding step not to analize the background, typical value=50 (also something Leica suggests automatically)
    space_mask = flim_plot.filters.make_space_mask_from_manual_threshold(image_1, threshold)

    # Generate G and S from the image that contains lifetime-dependent fluorophore 
    g, s, _ = flim_plot.phasor.get_phasor_components(image_1_time_sliced)

    # Calculate tau for each pixel
    phasor_tau_image_ps = np.zeros_like(g)
    #calculate tau in picoseconds where G is not zero and where space_mask is not zero
    phasor_tau_image_ps[g != 0 & space_mask] = (s[g != 0 & space_mask] / (2 * np.pi * laser_frequency * 1e6 * g[g != 0 & space_mask])) * 1e12 # in ps
    
    #processing intentity images and segmantation 
    #'Creating' intensity images 
    intensity_image_CFP = np.sum(image_1, axis=0)

    #based on the type of analisys ether 1 chanell segmentation, which is here, or 2 chanells 
    median_CFP, image4_elosrv1=segment_image(intensity_image_CFP)

    #Create a multitack image to evaluate paramets to measure features from labels
    multichannel_image = np.stack([intensity_image_CFP, g, s, phasor_tau_image_ps], axis=-1)

    #measure features from objects 
    properties = regionprops_table(label_image = image4_elosrv1, intensity_image = multichannel_image, properties = ['label', 'area','intensity_mean'])
    df = pd.DataFrame(properties)
    df.columns = ['label', 'area','average_fluor_intensity', 'average_G', 'average_S', 'average_phasor_tau_ps']

    #re-labeling the segmantation result with the values from the phasor analisys 
    param_image=relabel_with_map_array(image4_elosrv1, df['label'], df['average_phasor_tau_ps'])
    report_multichannel_image = np.stack([intensity_image_CFP, g, s, phasor_tau_image_ps, param_image], axis=-1)
    return df,report_multichannel_image

def analize_coloc_with_phasor(data_path_1,data_path_2,threshold = 50):
    """
    Analyzes FLIM data using phasor analysis from napri plugin napari_flim_phasor_plotter.
    Analizes only colocalizing objects, based on the segmantation of the intensity image. Which is created as a sum of counts in FLIM data 

    Args:
        data_path_1 (str): Path to the FLIM data file in ptu format 
        threshold (int, optional): Threshold value for background removal. Defaults to 50.

    Returns:
        tuple: Tuple containing:
            - pandas.DataFrame: Dataframe with features extracted from objects.
            - numpy.ndarray: Multichannel image with intensity, G, S, and phasor lifetime.
    """
        
    import napari_flim_phasor_plotter as flim_plot
    import numpy as np
    import napari
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from skimage.measure import label, regionprops_table
    import napari_simpleitk_image_processing as nsitk  # version 0.4.5
    import pyclesperanto_prototype as cle  # version 0.24.1
    import pandas as pd
    from napari_skimage_regionprops._parametric_images import relabel_with_map_array
    from FLIM_functions import segment_image
    
    #Loading PTU files and reading the images with correct chanells 
    list_layerdatatuple_1 = flim_plot._reader.read_single_ptu_file(data_path_1)
    list_layerdatatuple_2 = flim_plot._reader.read_single_ptu_file(data_path_2)

    list_layerdatatuple_1[0].shape
    list_layerdatatuple_2[0].shape

    # Get arrays and metadata for image1
    image1_channel0 = list_layerdatatuple_1[0][0]
    image1_channel1 = list_layerdatatuple_1[0][1]
    image1_metadata0 = list_layerdatatuple_1[1][0]
    image1_metadata1 = list_layerdatatuple_1[1][1]

    # Get arrays and metadata for image2
    image2_channel0 = list_layerdatatuple_2[0][0]
    image2_channel1 = list_layerdatatuple_2[0][1]
    image2_metadata0 = list_layerdatatuple_2[1][0]
    image2_metadata1 = list_layerdatatuple_2[1][1]

    # Conditional statements to select the needed channel
    if image1_channel0 is not None and np.any(image1_channel0): 
        image_1 = image1_channel0
        image_1_metadata = image1_metadata0
    else:
        image_1 = image1_channel1
        image_1_metadata = image1_metadata1

    if image2_channel0 is not None and np.any(image2_channel0): 
        image_2 = image2_channel0
    else:
        image_2 = image2_channel1

    # Get laset frequency from metadata
    laser_frequency = image_1_metadata['TTResult_SyncRate'] * 1E-6
    #print(f"Laser frequency: {laser_frequency} MHz")

    # Apply time mask (getting rid of the 'rising'part of the exponent from the entire image)
    time_mask = flim_plot.filters.make_time_mask(image_1, laser_frequency=laser_frequency)
    image_1_time_sliced = image_1[time_mask]

    #optional tresholding step not to analize the background, typical value=50 (also something Leica suggests automatically)
    space_mask = flim_plot.filters.make_space_mask_from_manual_threshold(image_1, threshold)

    # Generate G and S from the image that contains lifetime-dependent fluorophore 
    g, s, _ = flim_plot.phasor.get_phasor_components(image_1_time_sliced)

    # Calculate tau for each pixel
    phasor_tau_image_ps = np.zeros_like(g)
    #calculate tau in picoseconds where G is not zero and where space_mask is not zero
    phasor_tau_image_ps[g != 0 & space_mask] = (s[g != 0 & space_mask] / (2 * np.pi * laser_frequency * 1e6 * g[g != 0 & space_mask])) * 1e12 # in ps
    
    #processing intentity images and segmantation 
    #'Creating' intensity images 
    intensity_image_sir = np.sum(image_2, axis=0)
    intensity_image_CFP = np.sum(image_1, axis=0)

    #based on the type of analisys ether 1 chanell segmentation, which is here, or 2 chanells 
    median_CFP, image4_elosrv1=segment_image(intensity_image_sir)
    
    median_counts_cfp = nsitk.median_filter(intensity_image_CFP, 1, 1, 0) #but bin of 1 on sp8
    thb_counts_cfp = np.asarray(cle.threshold_otsu(median_counts_cfp)) 

    #create a mask with intersection of  objects from the segmentation with the second image 
    image2_ba = cle.binary_and(thb_counts_cfp,image4_elosrv1)
    objects_labels3 = label(image2_ba) # labeles intercection mask 

    #Create a multitack image to evaluate paramets to measure features from labels
    multichannel_image = np.stack([intensity_image_CFP, g, s, phasor_tau_image_ps], axis=-1)

    #measure features from objects 
    properties = regionprops_table(label_image = image4_elosrv1, intensity_image = multichannel_image, properties = ['label', 'area','intensity_mean'])
    df = pd.DataFrame(properties)
    df.columns = ['label', 'area','average_fluor_intensity', 'average_G', 'average_S', 'average_phasor_tau_ps']

    #re-labeling the segmantation result with the values from the phasor analisys 
    param_image=relabel_with_map_array(image4_elosrv1, df['label'], df['average_phasor_tau_ps'])
    report_multichannel_image = np.stack([intensity_image_CFP, g, s, phasor_tau_image_ps, param_image,intensity_image_sir], axis=-1)
    return df,report_multichannel_image

