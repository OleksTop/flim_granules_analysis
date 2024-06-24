
def in_folder_loop(global_path, selected_analysis_function, save_images=False, output_folder_path=None, date=None):
    import pandas as pd
    import natsort
    from pathlib import Path
    from FLIM_functions import analyze_single_fitted_channel, generate_small_report_fig, analyze_single_fastflim_channel
    from skimage.io import imsave
    """
    Process each pair of .tif files in the specified folder using the provided analysis function.

    Parameters
    ----------
    global_path : pathlib.Path
        The path to the folder containing the .tif files.
    selected_analysis_function : function
        The analysis function to apply to each pair of .tif files.
    save_images : bool, optional
        Whether to save the generated images (default is False).
    output_folder_path : pathlib.Path, optional
        The path to the folder where images will be saved (required if save_images is True).
    date : str, optional
        The date to include in the image filenames (required if save_images is True).

    Returns
    -------
    result_per_folder : pandas.DataFrame
        A DataFrame containing the results for each pair of images.
    report_per_folder : pandas.DataFrame
        A DataFrame containing the report data for each pair of images.

    Raises
    ------
    ValueError
        If save_images is True but output_folder_path or date is not provided.

    Examples
    --------
    >>> global_path = Path('path/to/folder')
    >>> selected_analysis_function = analyze_single_fitted_channel
    >>> result, report = in_folder_loop(global_path, selected_analysis_function, save_images=True, output_folder_path=Path('path/to/output'), date='2024-06-20')
    """

    result_per_folder = pd.DataFrame([])
    report_per_folder = pd.DataFrame([])

    # Here we build a paired_path nested list
    paired_path_list = []
    paired_paths = []

    # Create paired paths
    for file_path in natsort.natsorted(global_path.iterdir()):
        if file_path.suffix == '.tif': 
            paired_paths.append(file_path) 
            if len(paired_paths) == 2:
                paired_path_list.append(paired_paths)
                paired_paths = []

    # Process each pair of files in the current folder
    for paired_path in paired_path_list:
        result_per_image, report_per_image, multichannel_image = selected_analysis_function(paired_path[1], paired_path[0])

        # Add identification to the result table (image level)
        result_per_image['File Name'] = paired_path[1].stem
        report_per_image['File Name'] = paired_path[1].stem

        # Concatenate the filtered results to the local table
        result_per_folder = pd.concat([result_per_folder, result_per_image])
        report_per_folder = pd.concat([report_per_folder, report_per_image]) 

        # Start creating report images if save_images is True
        if save_images:
            if output_folder_path is None or date is None:
                raise ValueError("output_folder_path and date must be provided if save_images is True")

            # Construct the desired identification string for images
            identification = f"{date}_{paired_path[1].parent.name}_{paired_path[1].stem}"

            # Construct report images and save them
            small_report_fig = generate_small_report_fig(multichannel_image, identification, output_folder_path)

            # Save output images. Only 1 channel from param image (last channel)
            image_name = 'param_map_' + paired_path[1].stem + '.tif'
            imsave(output_folder_path / image_name, multichannel_image[..., -1])

    return result_per_folder, report_per_folder

# Define the Henderson-Hasselbalch model function
def hh_model(pH, pKa, tau_HA, tau_Aminus):
    """
    Calculate pH, based on the fluorescence lifetimes using the Henderson-Hasselbalch equation.
    The model assumes fluorophore is a weak acid and respective change in lifetimes reflects an acid-base titration 

    Parameters
    ----------
    pH : float or numpy.ndarray
        The pH value(s).
    pKa : float
        The pKa value.
    tau_HA : float
        The lifetime of the protonated form (HA).
    tau_Aminus : float
        The lifetime of the deprotonated form (A-).

    Returns
    -------
    float or numpy.ndarray
        The calculated pH value(s) from given lifetimes based on the Henderson-Hasselbalch equation.

    """
    # tau_meas = (tau_HA + tau_Aminus * 10 ** (pH - pKa)) / (1 + 10 ** (pH - pKa))
    ratio = 10 ** (pH - pKa)
    return (tau_HA + tau_Aminus * ratio) / (1 + ratio)
