@echo off
REM ============================================
REM RTX 4090 极致训练脚本 - AMSNetV2
REM 优化参数: batch-size=64, num-workers=8, AMP开启
REM ============================================

echo ===========================================
echo RTX 4090 AMSNetV2 Training Script
echo ===========================================

REM 设置Python环境 (如果需要)
REM call conda activate your_env_name

REM 设置CUDA优化
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM 切换到项目目录
cd /d %~dp0..

REM 检查参数
if "%1"=="" goto :usage
if "%1"=="dryrun" goto :dryrun
if "%1"=="quick" goto :quick
if "%1"=="full" goto :full
if "%1"=="ablation" goto :ablation
if "%1"=="baseline" goto :baseline
if "%1"=="profile" goto :profile
goto :usage

:dryrun
echo.
echo [MODE] Dry Run - Environment Validation
echo.
python DMC_Net_experiments.py ^
    --dataset dryrun ^
    --model amsv2 ^
    --batch-size 64 ^
    --epochs 2 ^
    --amp ^
    --num-workers 8 ^
    --profile
goto :end

:quick
echo.
echo [MODE] Quick LOSO Test (5 folds, 2 seeds, 50 epochs)
echo Estimated time: 2-3 hours
echo.
python DMC_Net_experiments.py ^
    --dataset sisfall ^
    --data-root ./data ^
    --model amsv2 ^
    --eval-mode loso ^
    --loso-max-folds 5 ^
    --seeds 42 123 ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-workers 8 ^
    --lr 1e-3 ^
    --weighted-loss ^
    --amp ^
    --use-tfcl ^
    --out-dir ./outputs/loso_quick_test
goto :end

:full
echo.
echo [MODE] Full LOSO Experiment (23 folds, 5 seeds, 100 epochs)
echo Estimated time: 24-36 hours
echo.
python DMC_Net_experiments.py ^
    --dataset sisfall ^
    --data-root ./data ^
    --model amsv2 ^
    --eval-mode loso ^
    --seeds 42 123 456 789 1024 ^
    --epochs 100 ^
    --batch-size 64 ^
    --num-workers 8 ^
    --lr 1e-3 ^
    --weight-decay 1e-4 ^
    --weighted-loss ^
    --amp ^
    --use-tfcl ^
    --out-dir ./outputs/loso_full_4090
goto :end

:ablation
echo.
echo [MODE] Ablation Study (7 configurations)
echo.

echo Running: Full Model...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp --use-tfcl ^
    --out-dir ./outputs/ablation/full

echo Running: w/o DKS...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp --use-tfcl ^
    --ablation dks=False --out-dir ./outputs/ablation/no_dks

echo Running: w/o MSPA...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp --use-tfcl ^
    --ablation mspa=False --out-dir ./outputs/ablation/no_mspa

echo Running: w/o FAA...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp --use-tfcl ^
    --ablation faa=False --out-dir ./outputs/ablation/no_faa

echo Running: w/o TFCL...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp ^
    --out-dir ./outputs/ablation/no_tfcl

echo Running: Time-only...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp ^
    --ablation time_only --out-dir ./outputs/ablation/time_only

echo Running: Freq-only...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model amsv2 ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp ^
    --ablation freq_only --out-dir ./outputs/ablation/freq_only

echo Ablation study completed!
goto :end

:baseline
echo.
echo [MODE] Baseline Comparison (5 models)
echo.

echo Running: LSTM...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model lstm ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 128 --num-workers 8 --amp ^
    --out-dir ./outputs/baselines/lstm

echo Running: ResNet1D...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model resnet ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 128 --num-workers 8 --amp ^
    --out-dir ./outputs/baselines/resnet

echo Running: TCN...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model tcn ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp ^
    --out-dir ./outputs/baselines/tcn

echo Running: Transformer...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model transformer ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 32 --num-workers 8 --amp ^
    --out-dir ./outputs/baselines/transformer

echo Running: InceptionTime...
python DMC_Net_experiments.py --dataset sisfall --data-root ./data --model inceptiontime ^
    --eval-mode loso --seeds 42 123 456 --epochs 100 --batch-size 64 --num-workers 8 --amp ^
    --out-dir ./outputs/baselines/inceptiontime

echo Baseline comparison completed!
goto :end

:profile
echo.
echo [MODE] Efficiency Profiling
echo.

echo Profiling: AMSNetV2...
python DMC_Net_experiments.py --dataset dryrun --model amsv2 --profile --epochs 1 --batch-size 1

echo Profiling: LSTM...
python DMC_Net_experiments.py --dataset dryrun --model lstm --profile --epochs 1 --batch-size 1

echo Profiling: ResNet1D...
python DMC_Net_experiments.py --dataset dryrun --model resnet --profile --epochs 1 --batch-size 1

echo Profiling: TCN...
python DMC_Net_experiments.py --dataset dryrun --model tcn --profile --epochs 1 --batch-size 1

echo Profiling: Transformer...
python DMC_Net_experiments.py --dataset dryrun --model transformer --profile --epochs 1 --batch-size 1

echo Profiling completed!
goto :end

:usage
echo.
echo Usage: train_4090.bat [mode]
echo.
echo Available modes:
echo   dryrun   - Quick environment validation (~1 min)
echo   quick    - Quick LOSO test (2-3 hours)
echo   full     - Full LOSO experiment for SCI paper (24-36 hours)
echo   ablation - Run all ablation experiments
echo   baseline - Run all baseline comparisons
echo   profile  - Run efficiency profiling
echo.
echo Example:
echo   train_4090.bat dryrun
echo   train_4090.bat full
echo.
echo Key Optimizations for RTX 4090:
echo   - batch-size: 64 (can increase to 128 for simpler models)
echo   - num-workers: 8 (adjust based on CPU cores)
echo   - AMP enabled (40%% speedup)
echo   - pin_memory enabled
echo.
goto :end

:end
echo.
echo ===========================================
echo Script finished.
echo ===========================================
pause
