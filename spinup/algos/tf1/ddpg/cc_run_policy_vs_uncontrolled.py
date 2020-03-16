import numpy as np
import re
from matplotlib import pyplot as plt
from pathlib import Path
from spinup.utils.test_policy import load_policy_and_env
from spinup.utils.logx import colorize
from seq2seq.utils import misc

# Disable GPU
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load saved environment and trained agent
p = Path(r'D:\chris\Documents\Programming\liveline_repos\ll_spinningup_clean\data\NEW')
env, get_action = load_policy_and_env(p)

# Reset environment with playhead at first batch (after optional trimming)
o = env.reset(playhead=env.trim_batches_start)

# Get target values for controlled outputs
tgts = np.array(list(env.lpp.data.output_setpoints.values()), dtype=o.dtype)

# Setup arrays to hold results
raw_array = list()
ctl_array = list()
tgt_array = list()
act_array = list()
nudge_array = list()

# Msg to console
print(colorize('\n\n' + '=' * 80, color='magenta', bold=False))
print(colorize('STARTING RUN\n', color='magenta', bold=False))


def image_saver(fig, tgt):
    """Saves plot figure as an image file.

    Arguments:
        fig (obj): A Matplotlib plot figure
        tgt (str): A valid save path
    """
    fig.savefig(tgt,
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
                dpi=150,
                transparent=False,
                )
    return


def list_of_nums_to_string(my_list):
    list_of_strings = [f'{format(elem, ".3f"):>6}' for elem in my_list]
    a_str = '[' + ', '.join(list_of_strings) + ']'
    return a_str


def improved(raw, obs, tgts):
    all_yes = True
    all_no = True
    improved_str = '\tbetter:\t['
    for r, o, t in zip(raw, obs, tgts):
        if np.abs(o - t) < np.abs(r - t):
            improved_str += '   YES, '
            all_no = False
        else:
            improved_str += '    NO, '
            all_yes = False
    improved_str = improved_str[:-2] + ']'
    return improved_str, all_yes, all_no


# Run simulation
for b in range(env.trim_batches_start, len(env.dataset_inputs) - env.trim_batches_end - 1):

    print(f'Batch: {b}\tplayhead: {env.playhead}')

    a = get_action(o)
    o, r, d, _ = env.step(a)
    raw_out = env.dataset_outputs[env.playhead][0]

    # Extract nudges from controllers
    n = list()
    for c in env.controllers:
        n.append(c.current_nudge)

    # Log results
    raw_array.append(raw_out)
    ctl_array.append(o)
    tgt_array.append(tgts)
    act_array.append(a)
    nudge_array.append(n)

    # Output to console
    print(f'\taction:\t{list_of_nums_to_string(a)}')
    print(f'\traw:\t{list_of_nums_to_string(raw_out)}')
    print(f'\tctrl:\t{list_of_nums_to_string(o)}')
    my_str, all_yes, all_no = improved(raw_out, o, tgts)

    if all_yes:
        # All outputs improved vs experimental data
        print(colorize(my_str, color='blue', bold=True))
    elif all_no:
        # All outputs worse than experimental data
        print(colorize(my_str, color='red', bold=True))
    else:
        # Mixed results - some better, some worse
        print(colorize(my_str, color='yellow', bold=False))
    print('')

    # Handle 'done' flag from env so we can keep on stepping sequentially
    env.returns['done'] = False if d else env.returns['done']

# Convert log arrays to Numpy
raw_array = np.array(raw_array, dtype=o.dtype)
ctl_array = np.array(ctl_array, dtype=o.dtype)
tgt_array = np.array(tgt_array, dtype=o.dtype)
act_array = np.array(act_array, dtype=o.dtype)
nudge_array = np.array(nudge_array, dtype=o.dtype)

# Calculate normalized RMSE (range)
# Note: raw_array was constructed using trimmed batches so it should not contain crazy end artifacts
# that would skew the min() and max().
rmse_raw = misc.rmse(raw_array, tgt_array) / (raw_array.max() - raw_array.min())
rmse_controlled = misc.rmse(ctl_array, tgt_array) / (raw_array.max() - raw_array.min())

# Save RMSE stats
f = 'rmse_stats.txt'
t = p / f
with open(t, 'w') as text_file:
    print('Normalized RMSE comparison (nrmse_range) versus output targets:', file=text_file)
    print(f'Uncontrolled: {rmse_raw}', file=text_file)
    print(f'Controlled:   {rmse_controlled}', file=text_file)


# ============================================================================
# Quick and dirty plotting: Output Sigs
# ============================================================================

# Pyplot color cycle v2.0
color_cycle = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
               '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', ]

x = range(raw_array.shape[0])
out_sig_names = env.lpp.data.transformed.outputs.columns.tolist()

# Figure loop
for idx, sig_name in enumerate(out_sig_names):

    # Color choice index
    c = color_cycle[idx % len(color_cycle)]

    # Figure
    fig = plt.figure(figsize=(8, 4.5))

    # Plots for this Figure
    plt.plot(x, raw_array[:, idx], color=c, linestyle='solid', linewidth=1.0, label='Uncontrolled', alpha=1.0)
    plt.plot(x, ctl_array[:, idx], color=c, linestyle='solid', linewidth=0.5, label='Controlled', alpha=0.50)

    # Decorate
    plt.title(f'{sig_name}')
    plt.xlabel('Time [s]')
    plt.ylabel('Std Dev')
    plt.legend()
    plt.show()

    # Save
    output_num = str([int(s) for s in re.findall(r'\d+', sig_name)][0])
    f = f'output_{output_num}' + '.png'
    t = p / f
    image_saver(fig, t)

    plt.close('all')

# ============================================================================
# Quick and dirty plotting: Controller actions, raw
# ============================================================================

points_to_plot = 2000

# Figure
fig = plt.figure(figsize=(8, 4.5))

# Controller loop
for idx in range(act_array.shape[1]):

    # Color choice index
    c = color_cycle[idx % len(color_cycle)]

    # Plot this controller
    plt.plot(x[0:points_to_plot], act_array[0:points_to_plot, idx],
             color=c, linestyle='solid', linewidth=1.0, label=f'Action {idx}', alpha=1.0)

# Decorate
plt.title(f'Actions, Raw Values')
plt.xlabel('Time [s]')
plt.ylabel('Action in Std Dev of Original Signal')
plt.legend()

plt.show()
plt.close('all')


# ============================================================================
# Quick and dirty plotting: Controller actions, moving avg
# ============================================================================

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


window = 50
act_ma = np.zeros_like(act_array)
for idx in range(act_array.shape[1]):
    act_ma[window - 1:, idx] = moving_average(act_array[:, idx], n=window)

# # Figure
# fig = plt.figure(figsize=(8, 4.5))
#
# # Plot
# for idx in range(act_ma.shape[1]):
#     # Color choice index
#     c = color_cycle[idx % len(color_cycle)]
#     plt.plot(x[0:2000], act_ma[0:2000, idx],
#              color=c, linestyle='solid', linewidth=1.0, label=f'Action {idx}', alpha=1.0)
#
# # Decorate
# plt.title(f'Actions - Moving Avg {window}')
# plt.xlabel('Time [s]')
# plt.ylabel('Action in Std Dev of Original Signal')
# plt.legend()
# plt.show()
# plt.close('all')

# ============================================================================
# Quick and dirty plotting: Controller actions, De-scaled (assumes StandardScaler)
# ============================================================================

# TODO: NO!!! We need env.lpp.data.raw.outputs and inputs!
# # Scaling parameters: For Outputs
# u_outputs = np.array(env.lpp.data.transformed.outputs.mean().tolist())
# s_outputs = np.array(env.lpp.data.transformed.outputs.std().tolist())
#
# # Scaling parameters: For Inputs and Actions
# u_inputs = np.array(env.lpp.data.transformed.inputs.mean().tolist())
# s_inputs = np.array(env.lpp.data.transformed.inputs.std().tolist())
idx_list = list()
for i, c in enumerate(env.controllers):
    all_input_sig_names = list(env.lpp.data.transformed.inputs)
    idx_list.append(all_input_sig_names.index(c.signal_name))
# u_inputs = u_inputs[idx_list]
# s_inputs = s_inputs[idx_list]

# TODO: Replace this hard-coding with the above
extruder_1_rpm_mean = 11.07209
extruder_1_rpm_std = 0.635172
extruder_2_rpm_mean = 13.8968
extruder_2_rpm_std = 0.029167
extruder_4_rpm_mean = 13.26966
extruder_4_rpm_std = 0.434594
u_inputs = np.array([extruder_1_rpm_mean, extruder_2_rpm_mean, extruder_4_rpm_mean])
s_inputs = np.array([extruder_1_rpm_std, extruder_2_rpm_std, extruder_4_rpm_std])

# Descale the actions; To return CHANGES in RPM, use mean = 0
act_array_descaled = (act_array * s_inputs) + 0
act_array_descaled_ma = np.zeros_like(act_array_descaled)
for idx in range(act_array_descaled.shape[1]):
    act_array_descaled_ma[window - 1:, idx] = moving_average(act_array_descaled[:, idx], n=window)

# Figure
fig = plt.figure(figsize=(8, 4.5))

# Plot
x = list(range(env.trim_batches_start, len(env.dataset_inputs) - env.trim_batches_end - 1))
for idx in range(act_array_descaled.shape[1]):
    # Color choice index
    c = color_cycle[idx % len(color_cycle)]
    plt.plot(x, act_array_descaled[:, idx],
             color=c, linestyle='solid', linewidth=1.0, alpha=0.5)
    plt.plot(x, act_array_descaled_ma[:, idx],
             color=c, linestyle='solid', linewidth=1.0, label=f'Extruder {idx + 1}', alpha=1.0)

# Decorate
plt.title(f'Actions, Descaled')
plt.xlabel('Time [s]')
plt.ylabel('Action in RPM Values')
plt.legend()
plt.show()

# Save
f = 'rpm_changes_vs_experiment.png'
t = p / f
image_saver(fig, t)
plt.close('all')

# ============================================================================
# Quick and dirty plotting: Controller raw input sigs, De-scaled (assumes StandardScaler)
# ============================================================================

# Get unbatched input data
inputs = env.lpp.data.transformed.inputs.to_numpy()                 # Get (scaled!) input signals
init_inputs = env.lpp.experiment_info.taxonomy['input_seq_len']
final_outputs = env.lpp.experiment_info.taxonomy['output_seq_len']
inputs = inputs[init_inputs:-final_outputs]                         # Allow for in/out seq batches

# Select controllable input signals
inputs = inputs[:, idx_list]

# Descale and trim the controllable input signals
inputs_descaled = (inputs * s_inputs) + u_inputs
inputs_descaled = inputs_descaled[env.trim_batches_start:-env.trim_batches_end-1, :]

# Descale the nudges; To return DELTA in RPMs, use mean = 0
nudge_array_descaled = (nudge_array * s_inputs) + 0
nudge_array_descaled_ma = np.zeros_like(nudge_array_descaled)
for idx in range(nudge_array_descaled.shape[1]):
    nudge_array_descaled_ma[window - 1:, idx] = moving_average(nudge_array_descaled[:, idx], n=window)

# Add nudges to inputs
inputs_descaled_w_nudge = inputs_descaled + nudge_array_descaled

# Figure
fig = plt.figure(figsize=(8, 4.5))

# Plot
for idx in range(inputs_descaled.shape[1]):

    # Color choice index
    c = color_cycle[idx % len(color_cycle)]

    # Plot original inputs
    plt.plot(x, inputs_descaled[:len(x), idx],
             color=c, linestyle='solid', linewidth=1.0, label=f'Extruder {idx}', alpha=1.0)

    # Plot inputs with nudges
    plt.plot(x, inputs_descaled_w_nudge[:len(x), idx],
             color=c, linestyle='solid', linewidth=1.0, alpha=0.5)

# Decorate
plt.title(f'Extruder RPM Settings\n(Solid = Experiment, Shaded = AI)')
plt.xlabel('Time [s]')
plt.ylabel('RPM')
plt.legend()
plt.show()

# Save
f = 'rpm_actual_values.png'
t = p / f
image_saver(fig, t)
plt.close('all')

# ============================================================================
# Workspace cleanup
# ============================================================================

plt.close('all')

# x = run_policy(env, get_action, render=False, max_ep_len=1000, num_episodes=10)

del a, all_no, all_yes, b, d, my_str, o, p, r, raw_out, tgts
del c, color_cycle, idx, x, window
del i, idx_list, all_input_sig_names
del act_array, act_array_descaled, act_array_descaled_ma, act_ma
del inputs, inputs_descaled, out_sig_names, points_to_plot, s_inputs, u_inputs
del extruder_1_rpm_mean, extruder_2_rpm_mean, extruder_4_rpm_mean
del extruder_1_rpm_std, extruder_2_rpm_std, extruder_4_rpm_std
del ctl_array, raw_array, tgt_array, sig_name, inputs_descaled_w_nudge
del f, fig, final_outputs, init_inputs, n, nudge_array, nudge_array_descaled, nudge_array_descaled_ma
del output_num, t, text_file
