## Step 1: Policy Extraction
python ep_plot.py --mode extract_policies --inputs ./exp_*.pkl --groupings exp --lastonly

## Step 2: Evaluations
python ../../../../ep_plot.py --mode single_group_metric --inputs exp_*.pkl.policy --evalmetric dgs --dataset pcb
python ../../../../ep_plot.py --mode single_group_metric --inputs exp_*.pkl.policy --evalmetric dpv --dataset pcb

## Step 3: Plotting
python ../../../../ep_plot.py --mode plot --inputs pcb.dgs --ylo -0.1 --yhi 1.1 --scale 0.1 --xlabel Episodes --ylabel "Goal Success" --plottitle "Goal Success" --saveplot goal_success.per_episode.pdf
python ../../../../ep_plot.py --mode plot --inputs pcb.dpv --ylo -0.1 --yhi 4 --scale 0.1 --xlabel Episodes --ylabel "Violations" --plottitle "Violations" --saveplot policy_violations.per_episode.pdf
