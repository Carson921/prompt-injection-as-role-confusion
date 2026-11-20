#!/usr/bin/env bash
# After both evolutions: IPI exact ranking, selection, panel P(user) of all conditions, ASR, then IPI-36 P(user) and the swap diagnostic.
cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo; export TOKENIZERS_PARALLELISM=false
while tmux -S /scratch/wz55191/tmux-richevo.sock has-session -t dsevo4ip 2>/dev/null || tmux -S /scratch/wz55191/tmux-richevo.sock has-session -t dschainA 2>/dev/null; do sleep 30; done
echo "IPI evolution and chainA ended $(date)"
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_finalize4.py outputs/dsevo4-20261119/ipi 8 > outputs/dsevo4-20261119/ipi/finalize.log 2>&1; echo "IPI finalize exit $? $(date)"
/scratch/wz55191/venvs/chatinject-table1/bin/python ds_select4.py outputs/dsevo4-20261119 outputs/optimized4-20261119 > outputs/optimized4-20261119/select.log 2>&1 || { echo "selection failed"; cat outputs/optimized4-20261119/select.log; exit 1; }
cat outputs/optimized4-20261119/select.log
PICKS=$(/scratch/wz55191/venvs/chatinject-table1/bin/python -c "import json;print(' '.join(json.load(open('outputs/optimized4-20261119/conditions.json'))))")
export DS_EXTRA_CONDITIONS=outputs/optimized4-20261119/conditions.json
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/agentdojo/dataset.json outputs/dsevo4-20261119/agentdojo/refs.json $PICKS >> outputs/dsevo4-20261119/refs.log 2>&1
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/ipi/dataset.json outputs/dsevo4-20261119/ipi/refs.json $PICKS ctrl_leadin opt2_top2 opt2_assistant original_delimiter original goal_only >> outputs/dsevo4-20261119/refs.log 2>&1
echo "panel refs done $(date)"
tmux -S /scratch/wz55191/tmux-richevo.sock kill-session -t dsprobe; sleep 20
bash asr2.sh outputs/optimized4-20261119 > outputs/optimized4-20261119/asr2.log 2>&1; echo "asr exit $? $(date)"
# native probing again: IPI-36 P(user) per behavior and the genuine-user-turn swap diagnostic
tmux -S /scratch/wz55191/tmux-richevo.sock new-session -d -s dsprobe "cd /scratch/wz55191/Github/prompt-injection-as-role-confusion/experiments/deepseek-richevo && export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/home/wz55191/.cache/triton && /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python -u ds_probe_server.py --output outputs/dsevo4-20261119/probe_server >> outputs/dsevo4-20261119/probe_server.log 2>&1"
until curl -sf -m 5 http://127.0.0.1:8791/health > /dev/null; do tmux -S /scratch/wz55191/tmux-richevo.sock has-session -t dsprobe || { echo "probe server died"; exit 1; }; sleep 20; done
echo "probe server ready $(date)"
/scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_diag_ipi_user.py outputs/dsevo4-20261119/joint_dataset.json outputs/dsevo4-20261119/diag_ipi_user.json > outputs/dsevo4-20261119/diag_ipi_user.log 2>&1; echo "diag exit $? $(date)"
DS_IPI_PRIMARY=1 /scratch/wz55191/venvs/chatinject-table1/bin/python -u ds_refs4.py outputs/dsevo4-20261119/ipi/dataset.json outputs/dsevo4-20261119/ipi36_refs.json true_user_turn delim_close_assistant_user $PICKS opt2_simple opt2_top1 opt_top1 delim_close_user opt2_top2 opt2_assistant ctrl_leadin original_delimiter original goal_only > outputs/dsevo4-20261119/ipi36_refs.log 2>&1
echo "ipi36 exit $? $(date)"
tmux -S /scratch/wz55191/tmux-richevo.sock kill-session -t dsprobe
echo "chainB done $(date)"
