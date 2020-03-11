# Paste into console:

import pickle
tgt = r'D:\chris\Documents\Programming\liveline_repos\ll_spinningup_clean\data\foo_experiment\foo_experiment_s42\replay_buffer.pkl'
p = pickle.dumps(replay_buffer)
with open(tgt, 'wb') as f:
    f.write(p)

import pickle
tgt = r'D:\chris\Documents\Programming\liveline_repos\ll_spinningup_clean\data\foo_experiment\foo_experiment_s42\replay_buffer.pkl'
with open(tgt, 'rb') as f:
    d = pickle.load(f)