from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils.logx import colorize
import gym
import liveline_gym
import numpy as np
from seq2seq.utils.misc import rmse
from matplotlib import pyplot as plt
from copy import deepcopy

# Disable GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load saved environment and trained agent
p = r'D:\chris\Documents\Programming\liveline_repos\ll_spinningup_clean\data\foo_experiment\foo_experiment_s42_5'
env, get_action = load_policy_and_env(p)

# Kill this later with refactored gym
env.dataset_inputs = env.dataset_inputs
env.dataset_outputs = env.lpp.data.batches.output_batches_overlapping

# Reset environment and get target values for controlled outputs
o = env.reset(playhead=env.trim_batches_start)
tgts = np.array(list(env.lpp.data.output_setpoints.values()), dtype=o.dtype)

# Setup arrays to hold results
raw_array = list()
ctl_array = list()
tgt_array = list()
act_array = list()

# Console
print(colorize('\n\n' + '='*80, color='magenta', bold=False))
print(colorize('STARTING RUN\n', color='magenta', bold=False))


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


for b, input_batch in enumerate(env.dataset_inputs):
    print(f'batch: {b}\tplayhead: {env.playhead}')
    a = get_action(o)
    o, r, d, _ = env.step(a)
    raw_out = env.dataset_outputs[b][0]

    # Log results
    raw_array.append(raw_out)
    ctl_array.append(o)
    tgt_array.append(tgts)
    act_array.append(a)

    # Output to console
    print(f'\taction:\t{list_of_nums_to_string(a)}')
    print(f'\traw:\t{list_of_nums_to_string(raw_out)}')
    print(f'\tctrl:\t{list_of_nums_to_string(o)}')
    my_str, all_y, all_n = improved(raw_out, o, tgts)
    if all_y:
        print(colorize(my_str, color='blue', bold=True))
    elif all_n:
        print(colorize(my_str, color='red', bold=True))
    else:
        print(colorize(my_str, color='yellow', bold=False))
    print('')

    # Handle 'done' from env so we can keep on stepping
    env.returns['done'] = False if d else env.returns['done']

    # if b > 50: break

    # NOTE: Playhead will wrap around if there is end trimming, as we've enumerated len of the dataset.

# Convert to Numpy
raw_array = np.array(raw_array, dtype=o.dtype)
ctl_array = np.array(ctl_array, dtype=o.dtype)
tgt_array = np.array(tgt_array, dtype=o.dtype)
act_array = np.array(act_array, dtype=o.dtype)

# Normalized RMSE (range)
rmse_raw = rmse(raw_array, tgt_array) / (raw_array.max() - raw_array.min())
rmse_controlled = rmse(ctl_array, tgt_array) / (raw_array.max() - raw_array.min())


# ============================================================================
# Quick and dirty plotting: Output Sigs
# ============================================================================

# Pyplot color cycle v2.0
color_cycle = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
               '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', ]

x = range(raw_array.shape[0])
out_sig_names = env.lpp.data.transformed.outputs.columns.tolist()

plt.close('all')

# Quick and dirty plot: By Output signal
for idx, sig_name in enumerate(out_sig_names):
    # Color choice index
    c = color_cycle[idx % len(color_cycle)]
    plt.plot(x, raw_array[:, idx], color=c, linestyle='solid', linewidth=1.0, label='Uncontrolled', alpha=1.0)
    plt.plot(x, ctl_array[:, idx], color=c, linestyle='solid', linewidth=0.5, label='Controlled', alpha=0.50)
    plt.title(f'{sig_name}')
    plt.xlabel('Time [s]')
    plt.ylabel('Std Dev')
    plt.legend()
    plt.show()

# ============================================================================
# Quick and dirty plotting: Controller actions, raw
# ============================================================================

# Controller actions
for idx in range(act_array.shape[1]):
    # Color choice index
    c = color_cycle[idx % len(color_cycle)]
    plt.plot(x[0:2000], act_array[0:2000, idx], color=c, linestyle='solid', linewidth=1.0, label=f'Action {idx}', alpha=1.0)
plt.title(f'Actions')
plt.xlabel('Time [s]')
plt.ylabel('Std Dev')
plt.legend()
plt.show()


# ============================================================================
# Quick and dirty plotting: Controller actions, moving avg
# ============================================================================

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


window = 50
act_ma = np.zeros_like(act_array)
for idx in range(act_array.shape[1]):
    act_ma[window-1:, idx] = moving_average(act_array[:, idx], n=window)


# Controller actions, smoothed
for idx in range(act_ma.shape[1]):
    # Color choice index
    c = color_cycle[idx % len(color_cycle)]
    plt.plot(x[0:2000], act_ma[0:2000, idx],
             color=c, linestyle='solid', linewidth=1.0, label=f'Action {idx}', alpha=1.0)
plt.title(f'Actions - Moving Avg {window}')
plt.xlabel('Time [s]')
plt.ylabel('Std Dev')
plt.legend()
plt.show()


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
    act_array_descaled_ma[window-1:, idx] = moving_average(act_array_descaled[:, idx], n=window)

# Controller actions, descaled, regular and moving avg
pts = 12000
for idx in range(act_array_descaled.shape[1]):
    # Color choice index
    c = color_cycle[idx % len(color_cycle)]
    plt.plot(x[0:pts], act_array_descaled[0:pts, idx],
             color=c, linestyle='solid', linewidth=1.0, alpha=0.5)
    plt.plot(x[0:pts], act_array_descaled_ma[0:pts, idx],
             color=c, linestyle='solid', linewidth=1.0, label=f'Extruder {idx+1}', alpha=1.0)
plt.title(f'Actions - Descaled')
plt.xlabel('Time [s]')
plt.ylabel('Change in RPM vs Original Experiment')
plt.legend()
plt.show()


# ============================================================================
# Quick and dirty plotting: Controller raw input sigs, De-scaled (assumes StandardScaler)
# ============================================================================

# Descale and select inputs
inputs = env.lpp.data.transformed.inputs.to_numpy()
inputs_descaled = (inputs[:, idx_list] * s_inputs) + u_inputs

delta_len = len(inputs_descaled) - len(act_array_descaled)

inputs_descaled_w_nudge = inputs_descaled[:-delta_len] + act_array_descaled

# Descaled inputs
for idx in range(inputs_descaled.shape[1]):
    # Color choice index
    c = color_cycle[idx % len(color_cycle)]
    plt.plot(x, inputs_descaled[:len(x), idx],
             color=c, linestyle='solid', linewidth=1.0, label=f'Extruder {idx}', alpha=1.0)
    plt.plot(x, inputs_descaled_w_nudge[:len(x), idx],
             color=c, linestyle='solid', linewidth=1.0, alpha=0.5)
plt.title(f'Extruder RPM Settings\n(Solid = Experiment, Shaded = AI)')
plt.xlabel('Time [s]')
plt.ylabel('RPM')
plt.legend()
plt.show()


# ============================================================================
# Workspace cleanup
# ============================================================================

plt.close('all')
# x = run_policy(env, get_action, render=False, max_ep_len=1000, num_episodes=10)
del a, all_n, all_y, b, d, input_batch, my_str, o, p, r, raw_out, tgts
del c, color_cycle, idx, x, window
del i, idx_list, all_input_sig_names
del act_array, act_array_descaled, act_array_descaled_ma, act_ma
del inputs, inputs_descaled, out_sig_names, pts, s_inputs, u_inputs
del extruder_1_rpm_mean, extruder_2_rpm_mean, extruder_4_rpm_mean
del extruder_1_rpm_std, extruder_2_rpm_std, extruder_4_rpm_std
del ctl_array, raw_array, tgt_array, sig_name, delta_len, inputs_descaled_w_nudge
