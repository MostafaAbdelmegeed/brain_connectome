@echo off
setlocal enabledelayedexpansion

rem List of run names
set run_names[0]=gcn_add_s0_ppmi_all_yeo17
set run_names[1]=gcn_add_s1_ppmi_all_yeo17
set run_names[2]=gcn_add_s2_ppmi_all_yeo17
set run_names[3]=gcn_add_s3_ppmi_all_yeo17
set run_names[4]=gcn_add_s4_ppmi_all_yeo17
set run_names[5]=gat_add_s0_ppmi_all_yeo17
set run_names[6]=gat_add_s1_ppmi_all_yeo17
set run_names[7]=gat_add_s2_ppmi_all_yeo17
set run_names[8]=gat_add_s3_ppmi_all_yeo17
set run_names[9]=gat_add_s4_ppmi_all_yeo17

rem List of commands to run
set cmds[0]=python -u polished/main.py --dataset ppmi --seed 0 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gcn --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gcn_add_s0_ppmi_all_yeo17
set cmds[1]=python -u polished/main.py --dataset ppmi --seed 1 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gcn --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gcn_add_s1_ppmi_all_yeo17
set cmds[2]=python -u polished/main.py --dataset ppmi --seed 2 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gcn --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gcn_add_s2_ppmi_all_yeo17
set cmds[3]=python -u polished/main.py --dataset ppmi --seed 3 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gcn --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gcn_add_s3_ppmi_all_yeo17
set cmds[4]=python -u polished/main.py --dataset ppmi --seed 4 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gcn --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gcn_add_s4_ppmi_all_yeo17
set cmds[5]=python -u polished/main.py --dataset ppmi --seed 0 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gat --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gat_add_s0_ppmi_all_yeo17
set cmds[6]=python -u polished/main.py --dataset ppmi --seed 1 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gat --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gat_add_s1_ppmi_all_yeo17
set cmds[7]=python -u polished/main.py --dataset ppmi --seed 2 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gat --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gat_add_s2_ppmi_all_yeo17
set cmds[8]=python -u polished/main.py --dataset ppmi --seed 3 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gat --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gat_add_s3_ppmi_all_yeo17
set cmds[9]=python -u polished/main.py --dataset ppmi --seed 4 --n_folds 10 --epochs 300 --patience 100 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --in_channels 58 --dropout 0.7 --model gat --gpu_id 0 --num_classes 4 --mode asym --network yeo17 --run_name gat_add_s4_ppmi_all_yeo17






rem Number of commands to run at a time
set max_parallel=3

rem Total number of commands
set total_cmds=10

set /a current_cmd=0

:run_next
rem Loop until all commands are run
if !current_cmd! lss %total_cmds% (
    rem Count running Python processes
    for /f "tokens=1" %%p in ('tasklist /FI "IMAGENAME eq python.exe" ^| find /C /I "python.exe"') do set num_running=%%p

    rem If fewer than 5 processes are running, start a new one
    if !num_running! lss %max_parallel% (
        start "" /B cmd /c !cmds[%current_cmd%]! > polished/logs/!run_names[%current_cmd%]!.log 2>&1
        set /a current_cmd+=1
        echo Starting new process !current_cmd!/!total_cmds!: !cmds[%current_cmd%]!
    )

    rem Wait for a short interval before checking again
    timeout /t 5 >nul

    rem Go back and check again
    goto run_next
)

echo All commands have been started.

:end
