#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=annavara_668
#SBATCH --job-name=agent4rec-rankcheck
#SBATCH --output=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec/reproduction_outputs/%x-%j.out
#SBATCH --error=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec/reproduction_outputs/%x-%j.err

set -euo pipefail

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
cd "$ROOT"
mkdir -p reproduction_outputs

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate
export PYTHONPATH="${ROOT}/.deps:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

python scripts/validate_checkpoint_rankings.py \
  --models MF LightGCN MultVAE \
  --num-users 1000 \
  --top-k 20 \
  --output reproduction_outputs/checkpoint_ranking_validation.json
