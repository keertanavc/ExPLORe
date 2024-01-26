python train_finetuning_deepsea.py \
    --env_name=DeepSea-bsuite \
    --max_steps=2000 \
    --log_interval=100\
    --eval_interval=100\
    --eval_episodes=10 \
    --project_name=deep_sea_run1 \
    --start_training=50\
    --seed=0


cd plotting
python wandb_dl.py --entity=vc-keertana --domain=deepsea --project_name=deep_sea_run1

import pickle as pl
import numpy as np

myfile = "deepsea.pkl"
with open(myfile, 'rb') as handle:
     my_array = pl.load(handle)
data = np.array(my_array)
