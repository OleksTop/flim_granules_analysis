
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
        result_per_image, report_per_image, multichannel_image = selected_analysis_function(paired_path[0], paired_path[1])

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


def sample_data_with_optional_balancing(data, n=100, balance_by_date=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    """
    Sample `n` instances for each combination of `transfection`, `cells`, and `treatment`.
    Optionally balance the sampling by `Date` if `balance_by_date` is True. Logs cases where
    fewer than `n` instances are available.

    Args:
    - data: DataFrame containing the data to sample from.
    - n: Number of instances to sample for each combination of `transfection`, `cells`, and `treatment`.
    - balance_by_date: Boolean flag indicating whether to balance samples by `Date` (default: False).

    Returns:
    - A tuple: (sampled_data, shortfall_log)
      sampled_data: A DataFrame with the sampled data.
      shortfall_log: A DataFrame logging groups where fewer than `n` instances were available.
    """
    # Group the data by transfection, cells, and treatment (ignoring Date for now)
    grouped_data = data.groupby(['transfection', 'cells', 'treatment'])

    # Initialize an empty DataFrame to store sampled data
    sampled_data = pd.DataFrame()

    # Initialize a list to log groups where fewer than `n` instances are available
    shortfall_log = []

    # Iterate over each group
    for group_name, group_df in grouped_data:
        total_available = len(group_df)

        # If the group has fewer than `n` instances, log the shortfall
        if total_available < n:
            shortfall_log.append({
                'group': group_name,
                'available_instances': total_available,
                'requested_samples': n
            })
            # Sample all available instances
            sampled_group_df = group_df
        else:
            # Shuffle the group_df to ensure randomness
            group_df = group_df.sample(frac=1, random_state=42)

            if balance_by_date:
                # Get the unique dates in this group
                unique_dates = group_df['Date'].unique()

                # If there's only one date, sample directly
                if len(unique_dates) == 1:
                    sampled_group_df = group_df.sample(n=n, replace=False)
                else:
                    # Calculate how many samples we want per date using proportional distribution
                    date_counts = group_df['Date'].value_counts()
                    samples_per_date = (date_counts / date_counts.sum() * n).astype(int)

                    # Initialize the sampled group DataFrame
                    sampled_group_df = pd.DataFrame()

                    # Take samples from each date based on the calculated proportional samples
                    for date, count in samples_per_date.items():
                        available_from_date = len(group_df[group_df['Date'] == date])
                        samples_to_take = min(count, available_from_date)
                        sampled_group_df = pd.concat([
                            sampled_group_df,
                            group_df[group_df['Date'] == date].sample(samples_to_take, replace=False)
                        ])

                    # Adjust to exactly `n` samples by adding/removing random samples if necessary
                    if len(sampled_group_df) < n:
                        additional_samples = group_df[~group_df.index.isin(sampled_group_df.index)].sample(n - len(sampled_group_df), replace=False)
                        sampled_group_df = pd.concat([sampled_group_df, additional_samples])
                    elif len(sampled_group_df) > n:
                        sampled_group_df = sampled_group_df.sample(n=n, replace=False)
            else:
                # If not balancing by date, just randomly sample `n` instances
                sampled_group_df = group_df.sample(n=n, replace=False, random_state=42)

        # Append the sampled data to the final DataFrame
        sampled_data = pd.concat([sampled_data, sampled_group_df], ignore_index=True)

    # Convert the shortfall log to a DataFrame for better readability
    shortfall_log_df = pd.DataFrame(shortfall_log)

    # Inspect the shortfall log
    if not shortfall_log_df.empty:
        print("Groups with fewer available instances than requested:")
        print(shortfall_log_df)
    else:
        print("No shortfalls found!")

    return sampled_data, shortfall_log_df

