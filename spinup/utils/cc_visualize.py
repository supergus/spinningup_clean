"""Visualization utilities.

Implements the following functions.
Most have options to print to console and/or save an image file to disk.

- `signals( )` : Plots input / output signals individually on separate Axes.
- `predictions( )` : Plots predicted versus observed outputs.
- `training( )` : Plots model training history.
- `scatter( )` : Plots 1 or 2 arrays with batch data. See docstring for details.
- `_image_saver( )` : Saves plots to disk as image files.
- `_pdf_saver( )` : Saves plots to disk as PDF files.

"""

__author__ = "Christopher Couch"
__license__ = "Strictly proprietary for Liveline Technologies, Inc."
__version__ = "2020-03"

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
import warnings
import datetime
import numpy as np
from scipy import signal
import sys
import pandas as pd
from matplotlib import ticker
from seq2seq.utils.paths import get_paths
from seq2seq.utils import misc
from keras.engine.training import Model

# Get path targets
paths = get_paths()
xformer_icon_tgt = paths['transformer_icon']
company_logo_tgt = paths['company_logo_for_dark_bkgd']


def _image_saver(f, tgt):
    """Saves plot figure as an image file.

    Arguments:
        f (obj): A Matplotlib plot figure
        tgt (str): A valid save path
    """
    f.savefig(tgt,
              facecolor=f.get_facecolor(),
              edgecolor=f.get_edgecolor(),
              dpi=600,
              transparent=False)
    return


def _pdf_saver(f, tgt):
    """Saves plot figure as a PDF file.

    Arguments:
        f (obj): A Matplotlib plot figure
        tgt (str): A valid save path
        tag (str): Additional text description
    """
    calling_function_name = sys._getframe(1).f_code.co_name

    # Multi-page PDF writer using a context manager
    with PdfPages(tgt) as pdf:
        # Page 1
        pdf.savefig(f,
                    facecolor=f.get_facecolor(),
                    edgecolor=f.get_edgecolor(),
                    dpi=600,
                    transparent=False)

        # Set the file's metadata
        d = pdf.infodict()
        d['Title'] = 'Liveline Signal Plots'
        d['Author'] = 'Christoper Couch'
        d['Subject'] = 'Generated from {}'.format(calling_function_name)
        d['Keywords'] = 'PdfPages Liveline signals'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

    return


def signals(settings=None, logger=None, data=None, results=None, verbosity=2, **kwargs):
    """Plots signals individually in separate figures with some nice formatting.

    Can be called with two methods:

    - Method 1: Using project objects, where pjt contains Settings, Logger, Data, and Results.
        Example: `visualize.signals(*pjt)`
    - Method 2: Using only a single DataFrame containing one or more signals.
        Example: `visualize.signals(df)`

    When calling with Method 2, the plot will be shown on the console, but will not
    be saved to disk. Certain plot features are also disabled such as showing a Project ID.

    |
    WARNING: If `tag` is not specified, default filenames will be used when saving plots.
    Those files will be OVER-WRITTEN if this function is called again without specifying
    a unique tag.

    Arguments:
        settings (obj): A valid Settings object.
        logger (obj): A valid Logger object.
        data (obj): A valid Data object. If not using, you must pass in a DataFrame using keyword arg `df`.
        results (obj): A valid Results object. Not currently used.
        verbosity (int): Optional. Default is 2. Valid choices are:
            3 - Print to console and save to disk.
            2 - No console, save to disk. Default value.
            1 - Console, no disk.
            0 - No console, no disk (why did you call the function? LOL).

    Keyword Arguments:
        sig (str): One of {'raw_inputs', 'raw_outputs', 'transformed_inputs', 'transformed_outputs'}.
            Selects DataFrame from data_obj. Required unless calling with Method 2.
        df (DataFrame): A Pandas DataFrame containing signal data with shape (samples, signals).
            If not using, you must pass in a valid Data object (Method 1).
        plotcols (int): Optional. How many columns of signal plots to display.
            If omitted, the script will try to find the best arrangement.
        data_range (tuple): Optional. Range of samples to display; a tuple of integers, (startval, endval);
            startval and endval are interpreted as sample numbers, not timestamps.
            If omitted, all available samples will be plotted.
        tag (str): Optional. Text to display in figure title and saved filename.
            If omitted, the contents of `str` will be used.
        save_to_exp (bool): Optional. If True, plots are saved in the /plots folder of the
            current experiment as defined in logger.current_experiment. If False, plots
            are saved in the project's /data_viz folder. Default is False.

    Returns:
        None

    """
    print('> Visualizing signals...')

    # Which calling method are we using?
    # If a DataFrame is supplied then it's Method 2 and we set method_1 = False
    df = kwargs.get('df', None)
    method_1 = True if df is None else False

    # Validate project objects and get parameters
    if method_1:
        misc.validate_objects(settings, logger, data, results)
        dataviz_folder = logger.project_info.dataviz_folder
        graphics_format = settings.visualization.graphics_format
        plotstyle = settings.visualization.plotstyle
        tracker = data.etl_tracker
    else:
        dataviz_folder = None
        graphics_format = None
        plotstyle = 'liveline'
        tracker = None
        # Validate DataFrame
        assert isinstance(df, pd.DataFrame), '`df` must be a valid Pandas DataFrame'

    # Parse other kwargs
    valid_kwargs = ['sig',
                    'df',
                    'plotcols',
                    'data_range',
                    'tag',
                    'save_to_exp',
                    ]

    for k in kwargs:
        assert k in valid_kwargs, 'Invalid keyword argument passed to `visualize.signals()`'

    sig = kwargs.get('sig', None)
    plotcols = kwargs.get('plotcols', None)
    data_range = kwargs.get('data_range', None)
    tag = kwargs.get('tag', None)
    save_to_exp = kwargs.get('save_to_exp', False)

    # Update save target to model folders (not project folders) if needed
    if method_1:
        if save_to_exp:
            idx = misc.parse_exp_index(logger, exp_id=logger.current_experiment)
            dataviz_folder = logger.experiment_info[idx]['paths'].plots_folder

    # Validate args
    assert isinstance(tag, str) or tag is None, '`tag` must be a valid string or None'
    if method_1 and sig is None:
        raise RuntimeError('Keyword `sig` is required to select correct signals from the Data object')

    # Select data from object
    if method_1:
        if sig == 'transformed_inputs':
            df = data.transformed.inputs
        elif sig == 'transformed_outputs':
            df = data.transformed.outputs
        elif sig == 'raw_inputs':
            df = data.raw.inputs
        elif sig == 'raw_outputs':
            df = data.raw.outputs
        else:
            raise RuntimeError('Cannot parse keyword `sig`')

    # Initialize
    plt.close('all')

    # Colors
    llred = (0.7569, 0.2392, 0.2392)
    llblue = (0, 0.4196, 0.5490)
    llyellow = (0.9921, 0.7333, 0.2314)

    # How many samples?
    n_samples = len(df)
    if n_samples < 2:
        raise RuntimeError('Not enough samples for visualization')

    # Get range of samples to visualize
    if data_range is None:
        data_range = (0, n_samples)
    else:
        # Convert tuple to list
        data_range = list(data_range)
        if data_range[1] == -1:
            data_range[1] = n_samples
        if data_range[1] > n_samples:
            data_range[1] = n_samples
            misc.print_warning('\tWARNING: `data_range` is invalid. Using last index in DataFrame as end point.')
        if data_range[0] < 0:
            data_range[0] = n_samples
            misc.print_warning('\tWARNING: `data_range` is invalid. Using 1st index in DataFrame as start point.')
    assert data_range[0] < data_range[1], 'Start of `data_range` must be less than end.'

    # Are we viewing all the data? Zoom or Full view
    full_view = True if (len(list(range(data_range[0], data_range[1]))) == n_samples) else False

    # Select plotcols
    n_plots = df.shape[1]
    if n_plots <= 2 and plotcols is None:
        plotcols = n_plots
    elif n_plots == 3 and plotcols is None:
        plotcols = 2
    elif n_plots > 3 and plotcols is None:
        plotcols = 4

    # Calculate rows & columns required; need 1 plot per signal
    if n_plots <= plotcols:
        # Need at least 1 row
        plotrows = 1
        plotcols = n_plots
    elif n_plots % plotcols:
        # Need an extra row that won't be full of plots
        plotrows = (n_plots // plotcols) + 1
    else:
        # All rows will be full of plots
        plotrows = n_plots // plotcols

    # Set style using a context manager
    with plt.style.context(plotstyle):

        # ========================================================
        # Setup the figure
        # ========================================================

        # Instantiate the figure
        f = plt.figure(figsize=((plotcols * 5.3333), (plotrows * 3)))  # 16:9 aspect ratio

        # Setup GridSpec
        g = gridspec.GridSpec(plotrows, plotcols, figure=f, )

        # Initialize list to hold Axes
        axlist = []

        # Main loop to generate plots
        for r in range(plotrows):
            for c in range(plotcols):

                # ========================================================
                # Create Axes and draw plot
                # ========================================================

                # Get position in subplot grid; we define pos=0 as the top left plot
                # and use `pos` to index the correct signal in the DataFrame
                pos = (r * plotcols) + c

                if pos < n_plots:

                    # Get data and create subplot
                    x = np.arange(data_range[0], data_range[1])
                    y = df.iloc[data_range[0]:data_range[1], pos:pos + 1]

                    # Create subplot Axes
                    ax = f.add_subplot(g[pos])

                    # Store Axes in a list
                    axlist.append(ax)

                    # Find line width to use (overrides plt.style)
                    lw = 0.8 if len(x) < 500 else 0.6

                    # Plot
                    axlist[pos].plot(x, y, color=llblue,
                                     linestyle='solid', linewidth=lw,
                                     label='{} {}'.format(tag, pos))  # alpha

                    # ========================================================
                    # Format axis labels
                    # ========================================================

                    # Suppress all y ticks
                    axlist[pos].get_yaxis().set_ticks([])

                    # Only show X axis labels and tick labels for subplots in bottom row
                    is_hanger = True if pos >= n_plots - plotcols else False
                    if (r == plotrows - 1) or is_hanger:
                        # Bottom row or a hanging plot; add x axis label
                        axlist[pos].set_xlabel('Samples')  # color, fontsize
                    else:
                        # Not bottom row or a hangar; suppress tick labels
                        axlist[pos].get_xaxis().set_ticks([])
                        pass

                    # ========================================================
                    # Add text box to Axes as "title" with signal name
                    # ========================================================

                    sig_name = y.columns.values[0]
                    axlist[pos].text(0.5, 1.075, '{}'.format(sig_name),
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     transform=axlist[pos].transAxes,
                                     color='k',
                                     bbox=dict(facecolor='w', alpha=0.5, boxstyle='round'),
                                     fontsize=8,
                                     fontweight='normal', )

                    # ========================================================
                    # Add horizontal lines to show mean and +/- 3 std dev
                    # ========================================================
                    # Calculate based on ALL available samples, not just selected range

                    all_data = df.iloc[:, pos]
                    mean = np.repeat(all_data.mean(), len(x))
                    three_sigma = np.repeat(all_data.std() * 3, len(x))

                    axlist[pos].plot(x, mean, color=llred, linestyle='dashed',
                                     linewidth=0.75, alpha=0.7, )
                    axlist[pos].plot(x, mean + three_sigma, color=llred,
                                     linestyle='solid', linewidth=0.75, alpha=1, )
                    axlist[pos].plot(x, mean - three_sigma, color=llred,
                                     linestyle='solid', linewidth=0.75, alpha=1, )

                    # ========================================================
                    # Add image / logo if the signal has been transformed and tracker available
                    # ========================================================
                    add_logo = False

                    if method_1 and sig[0:3] != 'raw':

                        if tracker is not None:
                            if sig_name in list(tracker.historian):
                                # See if we've done transformations on this signal
                                # by adding up the 1-hot vectors column-wise
                                if tracker.historian[[sig_name]].sum(axis=0)[0] > 0:
                                    add_logo = True
                            elif not tracker.historian.empty:
                                # sig_name is not in the tracker, so we've run a PCA
                                # and changed the input df columns to 'Component_1', 'Component_2', etc.
                                add_logo = True

                        if add_logo:
                            icon = mpimg.imread(str(xformer_icon_tgt))
                            imagebox = OffsetImage(icon, zoom=0.01, alpha=0.5)
                            xy = (0.976, 0.925)
                            ab = AnnotationBbox(imagebox, xy,
                                                xycoords='axes fraction',
                                                frameon=False,
                                                bboxprops=dict(facecolor='white', boxstyle='square', color='black')
                                                )
                            axlist[pos].add_artist(ab)

                    # ========================================================
                    # Other formatting / override plot style
                    # ========================================================

                    # Turn off grid lines
                    axlist[pos].grid(False)

                else:
                    # This grid position is blank; no signals to plot
                    pass

        # End of row, col main loop

        # Constrain the grid within a rectangular area on the figure
        g.tight_layout(f, rect=[0, 0, 1, .75], pad=1.0, )

        # ========================================================
        # Figure title
        # ========================================================

        y = 0.93

        if sig is not None and tag is None:
            s = sig.replace('_', ' ').title()
        if tag is not None:
            s = tag
        if sig is None and tag is None:
            s = 'Unknown Type'

        f.suptitle('Signals: {}'.format(s), y=y, fontsize=18, )

        # ========================================================
        # Add additional text: Subtitle
        # ========================================================

        if full_view:
            s = 'All available data shown'
        else:
            s = 'Partial view of data'

        # Add a text box, position relative to suptitle
        suptitle_y_pixels = f.bbox.ymax * f._suptitle._y
        y = (suptitle_y_pixels - 54) / f.bbox.ymax

        f.text(0.5, y, '{}\nWith means and +/- 3 sigma limits'.format(s),
               va='center', ha='center', color=llyellow, fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
               )

        # ========================================================
        # Add additional text: Experiment info
        # ========================================================

        if method_1:
            s = f'Project ID:\n{logger.project_info.project_id}'

            # Add a text box, position relative to figure
            y = (f.bbox.ymax - 48) / f.bbox.ymax

            f.text(0.99, y, f'{s}',
                   va='center', ha='right', color='w', fontsize=7,
                   fontweight='normal',
                   bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
                   )

        # ========================================================
        # Adding company logo
        # ========================================================

        if method_1:
            # Absolute positioning relative to figure
            y = (f.bbox.ymax - 48) / f.bbox.ymax
            x = (0 + 64) / f.bbox.xmax

            icon = mpimg.imread(str(company_logo_tgt))
            imagebox = OffsetImage(icon, zoom=0.175, alpha=1)
            xy = (x, y)
            ab = AnnotationBbox(imagebox, xy,
                                xycoords='figure fraction',
                                frameon=False,
                                bboxprops=dict(facecolor='white',
                                               boxstyle='square', color='black', alpha=.5)
                                )
            ax.add_artist(ab)

        # ========================================================
        # Print and save
        # ========================================================

        verbosity = 1 if not method_1 else verbosity

        # Save image file
        if verbosity == 2 or verbosity == 3:

            # Path for saving
            s = 'full_view' if full_view else 'partial_view'
            pre = sig if tag is None else tag.replace(' ', '_').lower()
            fname = '{}_signal_plots_{}'.format(pre, s)
            ext = '.' + graphics_format
            tgt = dataviz_folder / (fname + ext)

            # Save images
            if graphics_format != 'pdf':
                f = plt.gcf()
                _image_saver(f, tgt)

            if graphics_format == 'pdf':
                _pdf_saver(f, tgt)

        # Console output
        if verbosity == 1 or verbosity == 3:
            warnings.filterwarnings("ignore")  # Issues with PyCharm backend support
            plt.show()
            warnings.filterwarnings("default")

        # Cleanup
        plt.close('all')

    # End signals()
    return


def predictions(logger, settings, selection='batchwise', metric=None, batch_list=None, exp_id=None,
                index=None, tag=None, verbosity=2):
    """Plots best and worst predictions either batchwise or signalwise, based on RMSE.

    |
    Uses evaluation for most recent experiment_info in logger.

    Arguments:
        logger (obj): A valid Logger object. Contains save paths and Metrics object.
        settings (obj): A valid Settings object. Contains graphics format for saving.
        selection (str): Optional. Plots best/worst prediction either 'batchwise' or 'signalwise'.
            Default is 'batchwise.
        metric (str): Optional. One of {'nrmse_range', 'mase_baseline', 'mda'}.
            Default is settings.evaluation.metric.
        batch_list (list): Optional. Specific batches to plot. Batch indices correspond to those
            in the validation split. Indices must be a subset of
            logger.experiment_info[idx]['metrics'].batch_list.
            Default is to use all indices in logger.~.batch_list.
        exp_id (str): Optional. A valid experiment ID. If omitted, the current experiment
            will be targeted.
        index (int): Optional index to select an experiment from experiment_info dict in
            `logger.experiment_info`. If omitted and exp_id not supplied, the current experiment
            will be targeted. If provided, it will override exp_id.
        tag (str): For title and printing. Optional. Default is the most recent model name saved in logger.
        verbosity (int): Optional. Default value is 2.
            3 - Print to console and save plots.
            2 - No console, save plots. Default.
            1 - Console, no save.
    """
    # Validate
    misc.validate_objects(logger, settings)
    if selection not in ['batchwise', 'signalwise']:
        raise RuntimeError('`selection` must be either \'batchwise\' or \'signalwise\'')

    # Get ranking metric
    if metric is None:
        metric = settings.evaluation.metric
    if metric not in ['nrmse_range', 'mase_baseline', 'mda']:
        raise RuntimeError('`metric` must be either \'nrmse_range\', \'mase_baseline\', or \'mda\'')

    # Get experiment index
    idx = misc.parse_exp_index(logger, exp_id=exp_id, index=index)

    # Get batch indices for plotting (signalwise)
    full_list = logger.experiment_info[idx]['metrics'].batch_list
    full_list_indices = list(range(len(full_list)))
    if batch_list is None:
        batch_indices = full_list_indices
    else:
        batch_indices = [full_list.index(x) for x in batch_list]
        assert isinstance(batch_list, list), '`batch_list` must be a valid list'
        assert set(batch_list).issubset(set(full_list)), 'Your `batch_list` contains batches not used for predictions'

    # Get plot data
    inputs = logger.experiment_info[idx]['metrics'].inputs  # Numpy (batches, samples, signals)
    preds = logger.experiment_info[idx]['metrics'].predictions  # Numpy (batches, samples, signals)
    targets = logger.experiment_info[idx]['metrics'].targets  # Numpy (batches, samples, signals)
    assert targets.shape == preds.shape, \
        "Shapes of targets and predictions do not match"

    # Get tag or model name to use in titles etc.
    if tag is None:
        exp_id = logger.experiment_info[idx]['exp_id']
    else:
        exp_id = tag

    # Get paths, styles, and sig names
    dataviz_folder = logger.experiment_info[idx]['paths'].plots_folder
    graphics_format = settings.visualization.graphics_format
    plotstyle = settings.visualization.plotstyle
    target_sig_names = logger.experiment_info[idx]['signal_names']['transformed_outputs']

    # Get numbers of signals and samples.
    # Inputs and targets should have shape (batches, samples, signals).
    n_input_batches = inputs.shape[0]
    n_target_batches = targets.shape[0]
    n_target_samples = targets.shape[1]
    n_target_sigs = targets.shape[2]
    assert n_input_batches == n_target_batches, \
        "Numbers of batches for inputs and targets do not match"

    # Colors
    llred = (0.7569, 0.2392, 0.2392)
    llblue = (0, 0.4196, 0.5490)
    llyellow = (0.9921, 0.7333, 0.2314)

    # Set style using a context manager
    with plt.style.context(plotstyle):

        # Setup the figure
        f = plt.figure(figsize=(16 / 1.5, 9 / 1.5))

        # Setup GridSpec
        g = gridspec.GridSpec(1, 1, figure=f, )

        # Create subplot Axes
        ax1 = f.add_subplot(g[0])

        x = range(n_target_samples)

        # ========================================================
        # Batchwise plots
        # ========================================================

        if selection == 'batchwise':

            if metric == 'nrmse_range':
                best_b_idx = logger.experiment_info[idx]['metrics'].nrmse_range.flat.batchwise.best_at
                worst_b_idx = logger.experiment_info[idx]['metrics'].nrmse_range.flat.batchwise.worst_at
            if metric == 'mase_baseline':
                best_b_idx = logger.experiment_info[idx]['metrics'].mase_baseline.flat.batchwise.best_at
                worst_b_idx = logger.experiment_info[idx]['metrics'].mase_baseline.flat.batchwise.worst_at
            if metric == 'mda':
                best_b_idx = logger.experiment_info[idx]['metrics'].mda.flat.batchwise.best_at
                worst_b_idx = logger.experiment_info[idx]['metrics'].mda.flat.batchwise.worst_at

            p_best = preds[best_b_idx, :, :]  # (samples, signals)
            p_worst = preds[worst_b_idx, :, :]  # (samples, signals)
            t_best = targets[best_b_idx, :, :]  # (samples, signals)
            t_worst = targets[worst_b_idx, :, :]  # (samples, signals)

            # Plots with first signals & labels for legend
            # Best predictions, batchwise
            p1, = ax1.plot(x, p_best[:, 0], 'g-', alpha=1.0, linewidth=1.2, label=f'sigs from best batch {best_b_idx}')
            # Targets for best batches
            o1, = ax1.plot(x, t_best[:, 0], 'g:', alpha=0.5, linewidth=0.8, label='observed output sigs')
            # Worst predictions, batchwise
            p2, = ax1.plot(x, p_worst[:, 0], 'r-', alpha=0.75, linewidth=1.2,
                           label=f'sigs from worst batch {worst_b_idx}')
            # Targets for worst batches
            o2, = ax1.plot(x, t_worst[:, 0], 'r:', alpha=0.5, linewidth=0.8, label='observed output sigs')

            # Plot the other signals
            if n_target_sigs > 1:
                ax1.plot(x, p_best[:, 1:], 'g-', alpha=1.0, linewidth=1.2, )
                ax1.plot(x, t_best[:, 1:], 'g:', alpha=0.5, linewidth=0.8, )
                ax1.plot(x, p_worst[:, 1:], 'r-', alpha=0.75, linewidth=1.2, )
                ax1.plot(x, t_worst[:, 1:], 'r:', alpha=0.5, linewidth=0.8, )

        # ========================================================
        # Signalwise plots
        # ========================================================

        if selection == 'signalwise':

            if metric == 'nrmse_range':
                best_s_idx = logger.experiment_info[idx]['metrics'].nrmse_range.flat.signalwise.best_at
                worst_s_idx = logger.experiment_info[idx]['metrics'].nrmse_range.flat.signalwise.worst_at
            if metric == 'mase_baseline':
                best_s_idx = logger.experiment_info[idx]['metrics'].mase_baseline.flat.signalwise.best_at
                worst_s_idx = logger.experiment_info[idx]['metrics'].mase_baseline.flat.signalwise.worst_at
            if metric == 'mda':
                best_s_idx = logger.experiment_info[idx]['metrics'].mda.flat.signalwise.best_at
                worst_s_idx = logger.experiment_info[idx]['metrics'].mda.flat.signalwise.worst_at

            p_best = preds[:, :, best_s_idx]  # (batches, samples)
            p_worst = preds[:, :, worst_s_idx]  # (batches, samples)
            t_best = targets[:, :, best_s_idx]  # (batches, samples)
            t_worst = targets[:, :, worst_s_idx]  # (batches, samples)

            # Get output signal names
            osn_best = target_sig_names[best_s_idx]
            osn_worst = target_sig_names[worst_s_idx]

            # Plots with batches & labels for legend
            # Best predictions, signalwise, using first batch in batch_indices
            p1, = ax1.plot(x, p_best[0, :], 'g-', alpha=1.0, linewidth=1.2, label=f'best pred: {osn_best}')
            # Targets for best signal
            o1, = ax1.plot(x, t_best[0, :], 'g:', alpha=0.5, linewidth=0.8, label='observed output')
            # Worst predictions, signalwise, using first batch in batch_indices
            p2, = ax1.plot(x, p_worst[0, :], 'r-', alpha=0.75, linewidth=1.2, label=f'worst pred: {osn_worst}')
            # Targets for worst signal
            o2, = ax1.plot(x, t_worst[0, :], 'r:', alpha=0.5, linewidth=0.8, label='observed output')

            # Plot the other batches
            if len(batch_indices) > 1:
                for b in range(1, len(batch_indices)):
                    ax1.plot(x, p_best[batch_indices[b], :], 'g-', alpha=1.0, linewidth=1.2, )
                    ax1.plot(x, t_best[batch_indices[b], :], 'g:', alpha=0.5, linewidth=0.8, )
                    ax1.plot(x, p_worst[batch_indices[b], :], 'r-', alpha=0.75, linewidth=1.2, )
                    ax1.plot(x, t_worst[batch_indices[b], :], 'r:', alpha=0.5, linewidth=0.8, )

        # Legend
        leg = plt.legend(handles=[p1, o1, p2, o2], loc='upper right',
                         fontsize=6, facecolor='white', edgecolor='k', framealpha=1,
                         borderpad=1, ncol=2, columnspacing=6)

        # Legend text color
        plt.setp(leg.get_texts(), color='k')

        # Constrain the grid within a rectangular area on the figure
        g.tight_layout(f, rect=[0, 0, 1, .78], pad=1.0, )

        # ========================================================
        # Figure title
        # ========================================================

        y = 0.93
        f.suptitle(f'Predictions - {selection.capitalize()} Eval', y=y, fontsize=18, )

        # Text box
        if selection == 'batchwise':
            txt = f'Normalized Values, Best/Worst of {n_target_batches} Batches'
        if selection == 'signalwise':
            txt = f'Normalized Values, Best/Worst of {n_target_sigs} Output Signals, {len(batch_list)} Batches'
        f.text(0.5, 0.84, txt,
               va='center', ha='center', fontsize=12, fontweight='normal',
               color=llyellow,
               bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
               )

        # ========================================================
        # Add additional text: Experiment info
        # ========================================================

        s = f'Project and Experiment ID:\n{logger.project_info.project_id}\n{exp_id}'

        # Add a text box, position relative to figure
        y = (f.bbox.ymax - 48) / f.bbox.ymax

        f.text(0.99, y, f'{s}',
               va='center', ha='right', color='w', fontsize=7,
               fontweight='normal',
               bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
               )

        # ========================================================
        # Adding company logo
        # ========================================================

        # Absolute positioning relative to figure
        y = (f.bbox.ymax - 48) / f.bbox.ymax
        x = (0 + 64) / f.bbox.xmax

        icon = mpimg.imread(str(company_logo_tgt))
        imagebox = OffsetImage(icon, zoom=0.175, alpha=1)
        xy = (x, y)
        ab = AnnotationBbox(imagebox, xy,
                            xycoords='figure fraction',
                            frameon=False,
                            bboxprops=dict(facecolor='white',
                                           boxstyle='square', color='black', alpha=.5)
                            )
        ax1.add_artist(ab)

        # Reference line
        plt.axhline(0, color='black', linewidth=0.75, alpha=0.8)  # 0-line for reference

    # Console output
    if verbosity == 1 or verbosity == 3:
        warnings.filterwarnings("ignore")  # Issues with PyCharm backend support
        plt.show()
        warnings.filterwarnings("default")

    # Save figure
    if verbosity == 2 or verbosity == 3:
        fname = '{}_predictions_{}'.format(exp_id, selection)
        ext = '.' + graphics_format
        tgt = dataviz_folder / (fname + ext)

        if graphics_format != 'pdf':
            f = plt.gcf()
            _image_saver(f, tgt)

        if graphics_format == 'pdf':
            _pdf_saver(f, tgt)

    plt.close()

    # end of predictions()
    return


def training(h, logger, settings, exp_id=None, index=None, tag=None, verbosity=2):
    """Creates plots of training statistics.

    Saves on the path for most recent model info contained in `experiment_info[idx]['plots_folder']`.

    Arguments:
        h (obj):  Dict of metrics from a valid History object, either from Keras or a custom
            history implemented as a class extension.
        logger (obj): A valid Logger object. Contains save paths.
        settings (obj): A valid Settings object. Contains graphics formats for saving.
        exp_id (str): Optional. A valid experiment ID. If omitted, the current experiment
                will be targeted.
        index (int): Optional index to select an experiment from experiment_info dict in
                `logger.experiment_info`. If omitted and exp_id not supplied, the current experiment
                will be targeted. If provided, it will override exp_id.
        tag (str): For title and printing. Optional. Default is the most recent exp_id saved in logger.
        verbosity (int): Optional. Default is 2.
            3 - plots to console and saves image.
            2 - no plot to console, saves image. Default.
            1 - plot to console, does not save image.
            0 - no plot to console, no save.
    """

    misc.validate_objects(logger, settings)

    # Get experiment index
    idx = misc.parse_exp_index(logger, exp_id=exp_id, index=index)

    # Get values from history object; require validation metrics
    try:
        acc = h['acc']
        val_acc = h['val_acc']
        loss = h['loss']
        val_loss = h['val_loss']
    except KeyError:
        raise RuntimeError('One or more required training metrics are not available in history')

    # Get tag / exp name; tag overwrites exp_id
    if tag is None:
        exp_id = logger.experiment_info[idx]['exp_id']
    else:
        exp_id = tag

    # Other params
    dataviz_folder = logger.experiment_info[idx]['paths'].plots_folder
    graphics_format = settings.visualization.graphics_format
    plotstyle = settings.visualization.plotstyle

    # Set style using a context manager
    with plt.style.context(plotstyle):

        # Setup the figure
        f = plt.figure(figsize=(16 / 1.5, 9 / 1.5))

        # Setup GridSpec
        g = gridspec.GridSpec(1, 1, figure=f, )

        # Get x values for plotting (the epochs)
        # x = h.epoch
        x = list(range(1, len(acc) + 1))

        # Create subplot Axes
        ax1 = f.add_subplot(g[0])

        # Plots accuracy score on primary y axis (left)
        p1 = ax1.plot(x, acc, 'g-', alpha=1, label='Accuracy', linewidth=1.25)
        p2 = ax1.plot(x, val_acc, 'g--', alpha=0.8, label='Validation Accuracy', linewidth=0.75)

        # Gridlines - Tied only to primary y axis
        plt.grid(which='both', axis='both')
        # ax1.xaxis.grid(True) # Not needed

        # Legend
        leg1 = plt.legend(loc='upper right',
                          fontsize=6, facecolor='white', edgecolor='k',
                          framealpha=1, borderpad=1)
        plt.setp(leg1.get_texts(), color='k')

        # Plot loss score on secondary y axis (right)
        ax2 = ax1.twinx()
        p3 = ax2.plot(x, loss, 'r-', alpha=1, label='Loss', linewidth=1.25)
        p4 = ax2.plot(x, val_loss, 'r--', alpha=0.8, label='Validation Loss', linewidth=0.75)

        # Legend
        leg2 = plt.legend(loc='upper left',
                          fontsize=6, facecolor='white', edgecolor='k',
                          framealpha=1, borderpad=1)
        plt.setp(leg2.get_texts(), color='k')

        # Labels
        # plt.xlabel('Epoch')
        ax1.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax1.set_ylabel('Accuracy', fontsize=12, labelpad=8)
        ax2.set_ylabel('Loss', fontsize=12, labelpad=8)

        # Tick spacing and range: y axis
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(10))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(10))

        # Tick spacing and range: X axis
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(10, integer=True))
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(10, integer=True))

        # Gridlines - If created here, it will tie to secondary y axis
        # plt.grid(which='both', axis='both')

        # Constrain the grid within a rectangular area on the figure
        g.tight_layout(f, rect=[0, 0, 1, .85], pad=1.0, )

        # ========================================================
        # Figure title
        # ========================================================

        y = 0.93
        # f.suptitle('{} Training History'.format(exp_id.capitalize()), y=y, fontsize=18, )
        f.suptitle('Training History', y=y, fontsize=18, )

        # ========================================================
        # Add additional text: Experiment info
        # ========================================================

        s = f'Project and Experiment ID:\n{logger.project_info.project_id}\n{exp_id}'

        # Add a text box, position relative to figure
        y = (f.bbox.ymax - 48) / f.bbox.ymax

        f.text(0.99, y, f'{s}',
               va='center', ha='right', color='w', fontsize=7,
               fontweight='normal',
               bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
               )

        # ========================================================
        # Adding company logo
        # ========================================================

        # Absolute positioning relative to figure
        y = (f.bbox.ymax - 48) / f.bbox.ymax
        x = (0 + 64) / f.bbox.xmax
        icon = mpimg.imread(str(company_logo_tgt))
        imagebox = OffsetImage(icon, zoom=0.175, alpha=1)
        xy = (x, y)
        ab = AnnotationBbox(imagebox, xy,
                            xycoords='figure fraction',
                            frameon=False,
                            bboxprops=dict(facecolor='white',
                                           boxstyle='square', color='black', alpha=.5)
                            )
        ax1.add_artist(ab)

    if verbosity == 3 or verbosity == 1:
        warnings.filterwarnings("ignore")  # Issues with PyCharm backend support
        plt.show()
        warnings.filterwarnings("default")

    # Save figure
    if (dataviz_folder is not None) and (verbosity >= 2):

        fname = '{}_training_history.{}'.format(exp_id, graphics_format)
        tgt = dataviz_folder / fname

        if graphics_format != 'pdf':
            f = plt.gcf()
            _image_saver(f, tgt)

        if graphics_format == 'pdf':
            _pdf_saver(f, tgt)

    plt.close()

    # end of training()
    return


def scatter(array1, array2=None, logger=None, settings=None, exp_id=None, index=None, tag=None,
            array1_label=None, array2_label=None, verbosity=2):
    """Plots a scattergram using either 1 or 2 arrays.

    Unlike most other plot functions, this one is intended to be used after a project runs to
    manually explore the results. Therefore, the logger and settings arguments are not required.

    Args:
        array1 (obj): Required. An array of values with shape (). Accepts Python arrays, Numpy, or Pandas.
        array2 (obj): Optional. An array of values with shape (). Accepts Python arrays, Numpy, or Pandas.
            If provided, the shape must match array1.
        logger (obj): Optional. A valid Logger object. If omitted, saving to disk will be disabled.
        settings (obj): Optional. A valid Settings object. If omitted, plotstyle `liveline` and graphics
            format `png` will be used.
        exp_id (str): Optional. A valid experiment ID. If omitted, the current experiment
                will be targeted.
        index (int): Optional index to select an experiment from experiment_info dict in
                `logger.experiment_info`. If omitted and exp_id not supplied, the current experiment
                will be targeted. If provided, it will override exp_id.
        tag (str): For title and printing. Optional. Default is the most recent model name saved in logger.
        array1_label (str): Optional. Axis label for array 1. If omitted, 'Array 1 Values' will be used.
        array2_label (str): Optional. Axis label for array 2. If omitted, 'Array 2 Values' will be used.
        verbosity (int): Optional. Default is 2.
            3 - plots to console and saves image.
            2 - no plot to console, saves image. Default.
            1 - plot to console, does not save image.
            0 - no plot to console, no save.

    Returns:
        None.
    """

    # Get params
    if logger is not None:
        misc.validate_objects(logger)
        idx = misc.parse_exp_index(logger, exp_id=exp_id, index=index)
        exp_id = logger.experiment_info[idx]['exp_id']
        dataviz_folder = logger.experiment_info[idx]['paths'].plots_folder
        # Get tag / model_name for titles etc.
        if tag is not None:
            exp_id = tag

    else:
        idx = None
        dataviz_folder = None
        if tag is not None:
            exp_id = tag
        else:
            exp_id = 'default'
        if verbosity >= 2:
            misc.print_warning('Disabling save to disk - no Logger object provided')
            verbosity = 1  # Disable saving to disk

    if settings is not None:
        misc.validate_objects(settings)
        graphics_format = settings.visualization.graphics_format
        plotstyle = settings.visualization.plotstyle

    else:
        graphics_format = 'png'
        plotstyle = 'liveline'  # Will fail below if not installed!

    # Unbatch the first array
    array1 = misc.unbatch(misc.expander_3d(misc.arrays_to_numpy(array1)))

    # Unbatch the second array if we have one
    if array2 is not None:
        array2 = misc.unbatch(misc.expander_3d(misc.arrays_to_numpy(array2)))
        assert array2.shape == array1.shape, 'Arrays must have identical shapes'

    # Plot
    if array2 is None:
        # Plot array1 versus unit increments on x axis
        x_1d = np.expand_dims(np.array(list(range(len(array1)))), axis=1)
        x = np.repeat(x_1d, array1.shape[1], axis=1)
        y = array1
    else:
        # Plot array1 (x) vs array2 (y)
        x = array1
        y = array2

    # Setup for figure: X axis tick values and colors
    llred = (0.7569, 0.2392, 0.2392)
    llblue = (0, 0.4196, 0.5490)
    llyellow = (0.9921, 0.7333, 0.2314)
    # Pyplot color cycle v2.0
    color_cycle = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                   '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', ]

    # Initialize
    plt.close('all')

    with plt.style.context(plotstyle):

        # Setup the figure
        if array2 is None:
            f = plt.figure(figsize=(16 / 2, 9 / 2))  # 16:9 aspect ratio
        else:
            f = plt.figure(figsize=(16 / 2, 16 / 2))  # Square

        # Setup the GridSpec; one Axes per signal
        g = gridspec.GridSpec(1, 1, figure=f, )

        # Create subplot Axes in position 0
        ax = f.add_subplot(g[0])

        # For each signal (on a specific Axes), plot sparklines (batches) from each DataFrame,
        # with one color for each DataFrame
        for sig in range(array1.shape[1]):
            # ========================================================
            # Create Axes for this signal
            # ========================================================

            # Color choice index
            c = color_cycle[sig % len(color_cycle)]

            # Plot
            ax.scatter(x[:, sig], y[:, sig],
                       linestyle='solid', color=c,
                       label='Boy howdy!', alpha=0.15)

            # ========================================================
            # Format this signal Axes
            # ========================================================

            # Turn off grid lines
            ax.grid(False)

        # ========================================================
        # Axis labels and formatting
        # ========================================================

        if array2 is None:
            ax.set_xlabel('Samples', fontsize=12, labelpad=8)
            if array1_label is None:
                array1_label = 'Array Values'
            ax.set_ylabel(array1_label, fontsize=12, labelpad=8)
        else:
            if array1_label is None:
                array1_label = 'Array 1 Values'
            if array2_label is None:
                array2_label = 'Array 2 Values'
            ax.set_xlabel(array1_label, fontsize=12, labelpad=8)
            ax.set_ylabel(array2_label, fontsize=12, labelpad=8)

        # ========================================================
        # Constrain grid
        # ========================================================

        # Constrain the grid within a rectangular area on the figure
        g.tight_layout(f, rect=[0, 0, .985, .825], pad=1.0, )

        # ========================================================
        # Figure title
        # ========================================================

        y = 0.9525
        f.suptitle(f'{exp_id.capitalize()} Scattergram', y=y, fontsize=18, )

        # ========================================================
        # Adding company logo
        # ========================================================

        # Absolute positioning relative to figure
        y = (f.bbox.ymax - 42) / f.bbox.ymax
        x = (0 + 64) / f.bbox.xmax

        icon = mpimg.imread(str(company_logo_tgt))
        imagebox = OffsetImage(icon, zoom=0.175, alpha=1)
        xy = (x, y)
        ab = AnnotationBbox(imagebox, xy,
                            xycoords='figure fraction',
                            frameon=False,
                            bboxprops=dict(facecolor='white',
                                           boxstyle='square', color='black', alpha=.5)
                            )
        ax.add_artist(ab)

        # ========================================================
        # Add additional text: Experiment info
        # ========================================================

        if logger is not None:
            s = f'Project and Experiment ID:\n{logger.project_info.project_id}\n{exp_id}'

            # Add a text box, position relative to figure
            y = (f.bbox.ymax - 42) / f.bbox.ymax

            f.text(0.99, y, s,
                   va='center', ha='right', color='w', fontsize=7,
                   fontweight='normal',
                   bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
                   )

    # ========================================================
    # Save and show
    # ========================================================

    # Save figure
    if verbosity == 2 or verbosity == 3:
        fname = '{}_scattergram'.format(exp_id)
        pname = dataviz_folder
        ext = '.' + graphics_format
        tgt = pname / (fname + ext)

        if graphics_format != 'pdf':
            f = plt.gcf()
            _image_saver(f, tgt)

        if graphics_format == 'pdf':
            _pdf_saver(f, tgt)
        pass

    # Print to console
    if verbosity == 1 or verbosity == 3:
        warnings.filterwarnings("ignore")  # Issues with PyCharm backend support
        plt.show()
        warnings.filterwarnings("default")

    plt.close('all')

    # end of scatter()

    return


def n_ahead_preds(data, preds, batch_type=None, **kwargs):
    """Plots n-steps-ahead predictions against observed output values.

    Unlike most other plot functions, this one is intended to be used after a project runs to
    manually explore the results. Therefore, the logger and settings arguments are not required.

    Arguments:
        data (obj): Required. A valid Data object.
        preds (obj): Required. An array of predictions with shape (batches, samples, signals).
        batch_type (str): Required. One of {'adj', 'lap'}.

    Keyword Arguments:
        n (int or list): Optional. n-steps-ahead predictions to plot against observations.
            Default is to use the maximum steps ahead available in the
            predicted output sequences. If n is a list, all values in the list
            will be plotted on the same Axes.
        logger (obj): Optional. A valid Logger object. If omitted, saving to disk will be disabled.
        settings (obj): Optional. A valid Settings object. If omitted, plotstyle `liveline` and graphics
            format `png` will be used.
        signals (int or list): Optional. Indices of signals to plot. Default is to use all
            signals available. If signals is a list, all values in the list
            will be plotted on separate Axes.
        unscale (bool): Optional. If True, predictions and observations that were
            scaled for training will be returned to their original values.
        data_range (tuple): Optional. Range of samples to display; a tuple of integers,
            (startval, endval); startval and endval are interpreted as sample numbers,
            not timestamps. If omitted, all available samples will be plotted.
        exp_id (str): Optional. A valid experiment ID. If omitted, the current experiment
                will be targeted.
        index (int): Optional index to select an experiment from experiment_info dict in
                `logger.experiment_info`. If omitted and exp_id not supplied, the current experiment
                will be targeted. If provided, it will override exp_id.
        tag (str): Optional. For title and printing. Default is the most recent model name saved in logger.
        verbosity (int): Optional. Default is 2.
            3 - plots to console and saves image.
            2 - no plot to console, saves image. Default.
            1 - plot to console, does not save image.
            0 - no plot to console, no save.

    """

    # Validate kwargs
    valid_kwargs = ['n',
                    'logger',
                    'settings',
                    'signals',
                    'unscale',
                    'data_range',
                    'exp_id',
                    'index',
                    'tag',
                    'verbosity',
                    ]
    for k in kwargs:
        assert k in valid_kwargs, f'Keyword argument `{k}` not recognized'

    # Get kwargs
    n = kwargs.get('n', None)
    logger = kwargs.get('logger', None)
    settings = kwargs.get('settings', None)
    signals = kwargs.get('signals', None)
    unscale = kwargs.get('unscale', None)
    data_range = kwargs.get('data_range', None)
    exp_id = kwargs.get('exp_id', None)
    index = kwargs.get('index', None)
    tag = kwargs.get('tag', None)
    verbosity = kwargs.get('verbosity', 2)

    # Get params
    if logger is not None:
        misc.validate_objects(logger)
        idx = misc.parse_exp_index(logger, exp_id=exp_id, index=index)
        exp_id = logger.experiment_info[idx]['exp_id']
        dataviz_folder = logger.experiment_info[idx]['paths'].plots_folder
        # Get tag / model_name for titles etc.
        if tag is not None:
            exp_id = tag

    else:
        dataviz_folder = None
        if tag is not None:
            exp_id = tag
        else:
            exp_id = 'default'
        if verbosity >= 2:
            misc.print_warning('Disabling save to disk - no Logger object provided')
            verbosity = 1  # Disable saving to disk

    if settings is not None:
        misc.validate_objects(settings)
        graphics_format = settings.visualization.graphics_format
        plotstyle = settings.visualization.plotstyle

    else:
        graphics_format = 'png'
        plotstyle = 'liveline'

    # Get output observations
    misc.validate_objects(data)
    assert batch_type in ['lap', 'adj'], 'batch_type must be one of {\'lap\', \'adj\'}'
    if batch_type == 'lap':
        y_orig = data.batches.output_batches_overlapping.copy()
    else:
        y_orig = data.batches.output_batches_adjacent.copy()

    # Get signal names
    output_signal_names = data.transformed.outputs.columns.to_list()  # In order

    # Harvest some stats
    num_out_sigs = y_orig.shape[2]
    output_seq_len = y_orig.shape[1]

    # Get default value of n or else ensure it's a list
    if n is None:
        n = [output_seq_len]
    elif isinstance(n, int):
        n = [n]
    elif isinstance(n, list):
        pass

    # Validate entries in list n
    for elem in n:
        assert isinstance(elem, int), \
            'Values of n must be integers'
        assert (1 <= elem <= output_seq_len), \
            f'Values of n must be between 1 and output_seq_len ({output_seq_len}), inclusive'

    # Get default value of signals or else ensure it's a list
    only_one_signal = False
    all_available_signals = False
    multiple_selected_signals = False
    if isinstance(signals, list):
        multiple_selected_signals = True
    elif signals is None:
        signals = list(range(num_out_sigs))
        all_available_signals = True
    elif isinstance(signals, int):
        signals = [signals]
        only_one_signal = True

    # Validate entries in list s
    for elem in signals:
        assert isinstance(elem, int), \
            'Values of signals must be integers'
        assert (0 <= elem <= num_out_sigs - 1), \
            f'Values of `signals` must be between 0 and index of last available signal ({num_out_sigs - 1}), inclusive'

    # Unscale the preds and obs
    if unscale:
        obs_unscaled = misc.undo_output_scaling(data, y_orig.copy())
        preds_unscaled = misc.undo_output_scaling(data, preds.copy())
    else:
        obs_unscaled = y_orig.copy()
        preds_unscaled = preds.copy()

    # Unbatch the observations. We handle predictions in the plotting loop below.
    if batch_type == 'lap':
        obs_unscaled_unbatched = misc.unbatch(obs_unscaled, 'lap')
    else:
        obs_unscaled_unbatched = misc.unbatch(obs_unscaled, 'adj')

    # Get range of samples to visualize
    n_samples = len(obs_unscaled_unbatched)
    if data_range is None:
        data_range = (0, n_samples)
    else:
        # Convert tuple to list
        data_range = list(data_range)
        if data_range[1] == -1:
            data_range[1] = n_samples
        if data_range[1] > n_samples:
            data_range[1] = n_samples
            misc.print_warning('\tWARNING: `data_range` is invalid. Using last sample as end point.')
        if data_range[0] < 0:
            data_range[0] = n_samples
            misc.print_warning('\tWARNING: `data_range` is invalid. Using first sample as start point.')
    assert data_range[0] < data_range[1], 'Start of `data_range` must be less than end.'
    start = data_range[0]
    end = data_range[1] - 1

    # X-axis marks
    x_vals = range(len(obs_unscaled_unbatched))

    # Setup for figure: X axis tick values and colors
    llred = (0.7569, 0.2392, 0.2392)
    llblue = (0, 0.4196, 0.5490)
    llyellow = (0.9921, 0.7333, 0.2314)

    # Initialize
    plt.close('all')
    preds_to_plot = None
    elem = None

    # =================================================================================
    # Plot with area fill
    # =================================================================================
    with plt.style.context(plotstyle):

        # ========================================================
        # Setup figure, gridspec, Axes
        # ========================================================

        f = plt.figure(figsize=(16 / 2, 9 / 2))  # 16:9 aspect ratio

        # Setup the GridSpec; one Axes per signal
        g = gridspec.GridSpec(1, 1, figure=f, )

        # Create subplot Axes in position 0
        ax = f.add_subplot(g[0])

        for s in signals:

            for elem in n:
                # Extract n-ahead predictions from unscaled batches of preds,
                # pad start (first batch) with zeros before step n,
                # pad end (last batch) with remaining predictions after step n
                n_ahead = preds_unscaled[:, elem - 1, :]
                preds_to_plot = np.append(np.zeros((elem - 1, num_out_sigs)), n_ahead, axis=0)
                preds_to_plot = np.append(preds_to_plot, preds_unscaled[-1, elem - 1:-1, :], axis=0)

                # Plot predictions as a shaded area around the observations
                ax.fill_between(x_vals[start:end], preds_to_plot[start:end, s], obs_unscaled_unbatched[start:end, s],
                                alpha=0.66, color='r', linewidth=0)

                # # Plot preds as simple lines
                # ax.plot(x_vals[start:end], preds_to_plot[start:end, s], alpha=1.0, color='k', linewidth=0.75)

            # Plot observations
            ax.plot(x_vals, obs_unscaled_unbatched[:, s], color='k', alpha=1.0, linewidth=0.75)

            # ========================================================
            # Add additional text: Signal name
            # ========================================================

            # Add a text box, position relative to data coordinates

            y = preds_to_plot[start:end, s].mean()

            txt = f'{output_signal_names[s]} ({s})'
            ax.text(0, y, txt,
                    va='center', ha='left', color='k', fontsize=4,
                    fontweight='normal',
                    bbox=dict(facecolor='w', alpha=.80, boxstyle='round'),
                    )

        # ========================================================
        # Gridlines
        # ========================================================

        # Turn off grid lines
        ax.grid(True)

        # ========================================================
        # Axis labels and formatting
        # ========================================================

        ax.set_xlabel('Samples', fontsize=12, labelpad=8)
        if unscale:
            ax.set_ylabel('Output Values (Unscaled)', fontsize=12, labelpad=8)
        else:
            ax.set_ylabel('Output Values (Scaled)', fontsize=12, labelpad=8)

        # ========================================================
        # Constrain grid
        # ========================================================

        # Constrain the grid within a rectangular area on the figure
        g.tight_layout(f, rect=[0.015, 0, .985, .825], pad=1.0, )

        # ========================================================
        # Figure title
        # ========================================================

        y = 0.9525
        f.suptitle(f'{elem} Steps Ahead Predictions', y=y, fontsize=18, )

        # ========================================================
        # Adding company logo
        # ========================================================

        # Absolute positioning relative to figure
        y = (f.bbox.ymax - 42) / f.bbox.ymax
        x = (0 + 64) / f.bbox.xmax

        icon = mpimg.imread(str(company_logo_tgt))
        imagebox = OffsetImage(icon, zoom=0.175, alpha=1)
        xy = (x, y)
        ab = AnnotationBbox(imagebox, xy,
                            xycoords='figure fraction',
                            frameon=False,
                            bboxprops=dict(facecolor='white',
                                           boxstyle='square', color='black', alpha=.5)
                            )
        ax.add_artist(ab)

        # ========================================================
        # Add additional text: Project and Experiment info
        # ========================================================

        if logger is not None:
            t_str = f'Project and Experiment ID:\n{logger.project_info.project_id}\n{exp_id}'

            # Add a text box, position relative to figure
            y = (f.bbox.ymax - 42) / f.bbox.ymax

            f.text(0.97, y, t_str,
                   va='center', ha='right', color='w', fontsize=7,
                   fontweight='normal',
                   bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
                   )

        # ========================================================
        # Add additional text: Subtitle
        # ========================================================

        # Add a text box, position relative to suptitle
        suptitle_y_pixels = f.bbox.ymax * f._suptitle._y
        y = (suptitle_y_pixels - 54) / f.bbox.ymax

        if only_one_signal:
            txt = 'Selected Signal, Black=Observed, Red=Predicted'
        elif multiple_selected_signals:
            txt = 'Multiple Selected Signals, Black=Observed, Red=Predicted'
        elif all_available_signals:
            txt = 'All Available Signals, Black=Observed, Red=Predicted'
        else:
            txt = 'ERROR parsing signals'

        f.text(0.5, y, txt,
               va='center', ha='center', color=llyellow, fontsize=8,
               fontweight='normal',
               bbox=dict(facecolor='w', alpha=0, boxstyle='round'),
               )

        # ========================================================
        # Save and show
        # ========================================================

        # Save figure
        if verbosity == 2 or verbosity == 3:

            # Clean up
            steps = n[0] if len(n) == 1 else n

            if only_one_signal:
                fname = f'{exp_id}_{steps}_ahead_predictions_sig_{s}'
            elif multiple_selected_signals:
                fname = f'{exp_id}_{steps}_ahead_predictions_multiple_sigs'
            elif all_available_signals:
                fname = f'{exp_id}_{steps}_ahead_predictions_all_avail_sigs'

            pname = dataviz_folder
            ext = '.' + graphics_format
            tgt = pname / (fname + ext)

            if graphics_format != 'pdf':
                f = plt.gcf()
                _image_saver(f, tgt)

            if graphics_format == 'pdf':
                _pdf_saver(f, tgt)
            pass

        # Print to console
        if verbosity == 1 or verbosity == 3:
            warnings.filterwarnings("ignore")  # Issues with PyCharm backend support
            plt.show()
            warnings.filterwarnings("default")

        # End of signal loop; one plot per signal

    plt.close('all')

    # End of n_ahead_preds()
    return


if __name__ == "__main__":
    print("Howdy from `visualize.py`!")
else:
    pass
