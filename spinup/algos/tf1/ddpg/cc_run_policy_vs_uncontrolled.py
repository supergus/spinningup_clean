from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils.logx import colorize
import gym
import liveline_gym
import numpy as np
from seq2seq.utils.misc import rmse

# Load saved environment and trained agent
p = r'D:\chris\Documents\Programming\liveline_repos\ll_spinningup_clean\data\foo_experiment\foo_experiment_s42'
env, get_action = load_policy_and_env(p)

# Kill this later with refactored gym
env.dataset_inputs = env.dataset
env.dataset_outputs = env.lpp.data.batches.output_batches_overlapping

# Reset environment and get target values for controlled outputs
o = env.reset(playhead=env.trim_batches_start)
tgts = np.array(list(env.lpp.data.output_setpoints.values()), dtype=o.dtype)

# Setup arrays to hold results
raw_array = list()
ctl_array = list()
tgt_array = list()

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
    if b > 1000: break

# Convert to Numpy
raw_array = np.array(raw_array, dtype=o.dtype)
ctl_array = np.array(ctl_array, dtype=o.dtype)
tgt_array = np.array(tgt_array, dtype=o.dtype)

# RMSE
rmse_raw = rmse(raw_array, tgt_array)
rmse_controlled = rmse(ctl_array, tgt_array)


# x = run_policy(env, get_action, render=False, max_ep_len=1000, num_episodes=10)

del a, all_n, all_y, b, d, input_batch, my_str, o, p, r, raw_out, tgts
