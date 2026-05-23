@echo off
rem Windows wrapper for Isaac Lab ANYmal-D training (Isaac Sim binary install)
rem
rem Prerequisites:
rem   1. conda activate rl101-isaac
rem   2. Isaac Sim 5.1.x installed at C:\isaac-sim
rem   3. Directory junction: external\IsaacLab\_isaac_sim -> C:\isaac-sim
rem   4. flatdict pre-installed: pip install flatdict==4.0.1 --no-build-isolation
rem   5. Isaac Lab extensions installed manually (isaaclab.bat --install broken
rem      by flatdict/pkg_resources build-isolation bug on Windows).
rem
rem Usage:
rem   conda activate rl101-isaac
rem   E:\rl101-crash-course\scripts\train_anymal_win.bat
rem   E:\rl101-crash-course\scripts\train_anymal_win.bat --num_envs 1024

set PYTHONPATH=C:\isaac-sim\site
set ISAAC_PATH=C:\isaac-sim
set CARB_APP_PATH=C:\isaac-sim\kit
set EXP_PATH=C:\isaac-sim\apps
set RESOURCE_NAME=IsaacSim

cd /d E:\rl101-crash-course\external\IsaacLab

python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Rough-Anymal-D-v0 --headless %*
