#!/bin/bash
# Run all matchmaking experiments in parallel
# Usage: ./run_all_parallel.sh

echo "Starting all matchmaking experiments in parallel..."
echo "Logs will be saved to logs/ directory"

mkdir -p logs

# Run each experiment in background
python run_random.py > logs/random.log 2>&1 &
PID_RANDOM=$!
echo "Started Random (PID: $PID_RANDOM)"

python run_polarized.py > logs/polarized.log 2>&1 &
PID_POLARIZED=$!
echo "Started Polarized (PID: $PID_POLARIZED)"

python run_sbmm.py > logs/sbmm.log 2>&1 &
PID_SBMM=$!
echo "Started SBMM (PID: $PID_SBMM)"

python run_diverse.py > logs/diverse.log 2>&1 &
PID_DIVERSE=$!
echo "Started Diverse (PID: $PID_DIVERSE)"

echo ""
echo "All experiments started. PIDs:"
echo "  Random:    $PID_RANDOM"
echo "  Polarized: $PID_POLARIZED"
echo "  SBMM:      $PID_SBMM"
echo "  Diverse:   $PID_DIVERSE"
echo ""
echo "Monitor with: tail -f logs/*.log"
echo "Check wandb: https://wandb.ai/carlo-romeo-alt427/ARC"
echo ""

# Wait for all to complete
wait $PID_RANDOM $PID_POLARIZED $PID_SBMM $PID_DIVERSE

echo "All experiments complete!"
