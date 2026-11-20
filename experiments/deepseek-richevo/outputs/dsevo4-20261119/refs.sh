#!/usr/bin/env bash
# Reference measurements with the corrected probe (exact full forwards), in priority order.
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo
export TOKENIZERS_PARALLELISM=false
K="true_user_turn delim_close_assistant_user opt2_simple opt_top1 opt2_top1 delim_close_user"
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/joint_dataset.json outputs/dsevo4-20261119/refs_joint.json $K goal_only original
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/agentdojo/dataset.json outputs/dsevo4-20261119/agentdojo/refs.json $K
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/ipi/dataset.json outputs/dsevo4-20261119/ipi/refs.json $K
echo "refs stage1 done $(date)"
