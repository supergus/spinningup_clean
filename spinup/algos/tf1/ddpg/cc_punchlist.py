
# Check controller and env modes. OK? Grid search?
# Open up controller limits in env, just use action limits in main script
# Evaluate
# Try another algo? TD3, SAC

# TODO: With controller limits "wide open" we see controller outputs like +/- 32 !!! With nudges like 0.8.
#  How can that be when we're in absolute controller mode?

# TODO: Does everything run faster on CPU or GPU?

# TODO: When saving config JSON, also save env.info() so we can audit environment parameters.

# TODO: Saving of model... is it overwriting each epoch or only if "improved" performance?

# TODO: Create method to read config JSON & env.info() & controller info & epochs elapsed & load model, then
#  resume training. Important - with 20+ hrs of training time, you WILL lose power etc.
#  Can we also load progress through GridSearch?

# TODO: Add another metric for logging... nrmse_range versus target output values for (1) base dataset and
#  (2) controlled output. Pre-select B random batches (~250? see model pkg) and use the same batches for
#  (1) and (2). Are the controls helping us run "tighter"?


