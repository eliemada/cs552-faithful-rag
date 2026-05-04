#!/bin/bash
# CS-552 Project — RCP Job Launcher
# This script submits an interactive RCP job for running individual notebooks.
# Usage: ./submit.sh
#
# Prerequisites:
#   - RCP cluster access configured
#   - Python environment with project dependencies
#
# [MORE DETAILS TO COME from course staff]

# TODO: Update with actual RCP submission command once provided
# Example structure:
# srun --partition=gpu \
#      --gres=gpu:1 \
#      --mem=32G \
#      --time=04:00:00 \
#      --pty jupyter lab --no-browser --port=8888

echo "CS-552 Faithful RAG — Notebook Environment"
echo "============================================"
echo ""
echo "Waiting for RCP submission details from course staff."
echo "In the meantime, you can run notebooks locally with:"
echo "  pip install -r requirements.txt"
echo "  jupyter lab notebooks/"
