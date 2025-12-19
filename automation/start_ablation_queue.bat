@echo off
REM 消融实验队列管理脚本启动器
REM Ablation Experiment Queue Manager Launcher

echo ================================================================================
echo 消融实验队列管理脚本 (Ablation Experiment Queue Manager)
echo ================================================================================
echo.
echo 功能说明:
echo   - 监控 6 个 baseline 模型训练状态
echo   - 当 2 个 baseline 完成时，启动第 1 个 ablation 实验
echo   - 之后每完成 1 个 baseline，启动 1 个 ablation 实验
echo   - 总共启动 6 个 ablation 实验（按优先级排序）
echo.
echo Features:
echo   - Monitors 6 baseline models training status
echo   - Starts 1st ablation when 2 baselines complete
echo   - Starts 1 new ablation for each additional baseline completion
echo   - Total: 6 ablation experiments (prioritized)
echo.
echo ================================================================================
echo.
echo 消融实验列表 (Ablation Experiments):
echo   1. No_DKS       - 禁用动态卷积核选择 (Priority 1, Most Important)
echo   2. No_FAA       - 禁用 Fall-Aware Attention (Priority 2)
echo   3. No_MSPA      - 禁用多尺度频谱金字塔 (Priority 3)
echo   4. No_TFCL      - 禁用时频对比学习 (Priority 4)
echo   5. Time_Only    - 仅时域分支 (Priority 5)
echo   6. Freq_Only    - 仅频域分支 (Priority 6)
echo.
echo ================================================================================
echo.

REM 检查 Python 脚本是否存在
if not exist ablation_queue_manager.py (
    echo 错误: ablation_queue_manager.py 未找到！
    echo Error: ablation_queue_manager.py not found!
    pause
    exit /b 1
)

REM 激活虚拟环境（如果有的话）
if exist venv\Scripts\activate.bat (
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
)

REM 启动队列管理脚本
echo 启动队列管理脚本...
echo Starting queue manager...
echo.
python ablation_queue_manager.py

echo.
echo ================================================================================
echo 队列管理脚本已退出 (Queue Manager Exited)
echo ================================================================================
pause
