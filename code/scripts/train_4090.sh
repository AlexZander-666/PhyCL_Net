#!/bin/bash
# ============================================
# RTX 4090 极致训练脚本 - AMSNetV2
# ============================================

echo "==========================================="
echo "RTX 4090 AMSNetV2 Training Script"
echo "==========================================="

# 设置CUDA优化
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 切换到项目目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

case "$1" in
    dryrun)
        echo -e "\n${GREEN}[MODE] Dry Run - Environment Validation${NC}\n"
        python DMC_Net_experiments.py \
            --dataset dryrun \
            --model amsv2 \
            --batch-size 64 \
            --epochs 2 \
            --amp \
            --profile
        ;;

    quick)
        echo -e "\n${GREEN}[MODE] Quick LOSO Test (5 folds, 2 seeds, 50 epochs)${NC}"
        echo -e "${YELLOW}Estimated time: 2-3 hours${NC}\n"
        python DMC_Net_experiments.py \
            --dataset sisfall \
            --data-root ./data \
            --model amsv2 \
            --eval-mode loso \
            --loso-max-folds 5 \
            --seeds 42 123 \
            --epochs 50 \
            --batch-size 64 \
            --lr 1e-3 \
            --weighted-loss \
            --amp \
            --use-tfcl \
            --out-dir ./outputs/loso_quick_test
        ;;

    full)
        echo -e "\n${GREEN}[MODE] Full LOSO Experiment (23 folds, 5 seeds, 100 epochs)${NC}"
        echo -e "${YELLOW}Estimated time: 24-36 hours${NC}\n"
        python DMC_Net_experiments.py \
            --dataset sisfall \
            --data-root ./data \
            --model amsv2 \
            --eval-mode loso \
            --seeds 42 123 456 789 1024 \
            --epochs 100 \
            --batch-size 64 \
            --lr 1e-3 \
            --weight-decay 1e-4 \
            --weighted-loss \
            --amp \
            --use-tfcl \
            --out-dir ./outputs/loso_full_4090
        ;;

    full-bg)
        echo -e "\n${GREEN}[MODE] Full LOSO Experiment (Background Mode)${NC}"
        echo -e "${YELLOW}Running in background with nohup...${NC}\n"
        mkdir -p ./outputs/loso_full_4090
        nohup python DMC_Net_experiments.py \
            --dataset sisfall \
            --data-root ./data \
            --model amsv2 \
            --eval-mode loso \
            --seeds 42 123 456 789 1024 \
            --epochs 100 \
            --batch-size 64 \
            --lr 1e-3 \
            --weight-decay 1e-4 \
            --weighted-loss \
            --amp \
            --use-tfcl \
            --out-dir ./outputs/loso_full_4090 \
            > ./outputs/loso_full_4090/train.log 2>&1 &
        echo -e "${GREEN}Training started in background. PID: $!${NC}"
        echo "Monitor with: tail -f ./outputs/loso_full_4090/train.log"
        ;;

    ablation)
        echo -e "\n${GREEN}[MODE] Ablation Study (7 configurations)${NC}\n"

        configs=(
            "full::"
            "no_dks:--ablation dks=False"
            "no_mspa:--ablation mspa=False"
            "no_faa:--ablation faa=False"
            "no_tfcl:"
            "time_only:--ablation time_only"
            "freq_only:--ablation freq_only"
        )

        for config in "${configs[@]}"; do
            name="${config%%:*}"
            args="${config#*:}"

            echo -e "${YELLOW}Running: $name...${NC}"

            if [[ "$name" == "no_tfcl" ]]; then
                python DMC_Net_experiments.py \
                    --dataset sisfall --data-root ./data --model amsv2 \
                    --eval-mode loso --seeds 42 123 456 --epochs 100 \
                    --batch-size 64 --amp \
                    --out-dir ./outputs/ablation/$name
            else
                python DMC_Net_experiments.py \
                    --dataset sisfall --data-root ./data --model amsv2 \
                    --eval-mode loso --seeds 42 123 456 --epochs 100 \
                    --batch-size 64 --amp --use-tfcl \
                    $args --out-dir ./outputs/ablation/$name
            fi
        done

        echo -e "${GREEN}Ablation study completed!${NC}"
        ;;

    baseline)
        echo -e "\n${GREEN}[MODE] Baseline Comparison (5 models)${NC}\n"

        declare -A models=(
            ["lstm"]=128
            ["resnet"]=128
            ["tcn"]=64
            ["transformer"]=32
            ["inceptiontime"]=64
        )

        for model in "${!models[@]}"; do
            bs=${models[$model]}
            echo -e "${YELLOW}Running: $model (batch_size=$bs)...${NC}"
            python DMC_Net_experiments.py \
                --dataset sisfall --data-root ./data --model $model \
                --eval-mode loso --seeds 42 123 456 --epochs 100 \
                --batch-size $bs --amp \
                --out-dir ./outputs/baselines/$model
        done

        echo -e "${GREEN}Baseline comparison completed!${NC}"
        ;;

    profile)
        echo -e "\n${GREEN}[MODE] Efficiency Profiling${NC}\n"

        for model in amsv2 lstm resnet tcn transformer inceptiontime; do
            echo -e "${YELLOW}Profiling: $model${NC}"
            python DMC_Net_experiments.py \
                --dataset dryrun --model $model \
                --profile --epochs 1 --batch-size 1
            echo ""
        done
        ;;

    monitor)
        echo -e "\n${GREEN}[MODE] Training Monitor${NC}\n"
        if [ -f "./outputs/loso_full_4090/experiment.log" ]; then
            tail -f ./outputs/loso_full_4090/experiment.log
        else
            echo -e "${RED}Log file not found. Start training first.${NC}"
        fi
        ;;

    gpu)
        echo -e "\n${GREEN}[MODE] GPU Monitor${NC}\n"
        watch -n 1 nvidia-smi
        ;;

    *)
        echo ""
        echo "Usage: ./train_4090.sh [mode]"
        echo ""
        echo "Available modes:"
        echo "  dryrun   - Quick environment validation (~1 min)"
        echo "  quick    - Quick LOSO test (2-3 hours)"
        echo "  full     - Full LOSO experiment for SCI paper (24-36 hours)"
        echo "  full-bg  - Full LOSO in background (with nohup)"
        echo "  ablation - Run all ablation experiments"
        echo "  baseline - Run all baseline comparisons"
        echo "  profile  - Run efficiency profiling"
        echo "  monitor  - Monitor training log"
        echo "  gpu      - Monitor GPU usage"
        echo ""
        echo "Examples:"
        echo "  ./train_4090.sh dryrun"
        echo "  ./train_4090.sh full-bg"
        echo ""
        ;;
esac

echo ""
echo "==========================================="
echo "Script finished."
echo "==========================================="
