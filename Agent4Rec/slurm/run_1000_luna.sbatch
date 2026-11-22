#!/bin/bash
#SBATCH --job-name=agent4rec-luna-1000
#SBATCH --account=annavara_668
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=reproduction_outputs/agent4rec-luna-1000-%j.out
#SBATCH --error=reproduction_outputs/agent4rec-luna-1000-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
cd "${ROOT}"
export PYTHONPATH="${ROOT}/.deps:${ROOT}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-agent4rec"
export PYTHONUNBUFFERED=1

if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not available" >&2
    exit 2
fi

python scripts/check_luna_access.py
python main.py \
    --simulation_name agent4rec_lightgcn_luna_1000_5pages \
    --dataset ml-1m \
    --modeltype LightGCN \
    --model_path Saved \
    --n_avatars 1000 \
    --items_per_page 4 \
    --max_pages 5 \
    --execution_mode parallel \
    --max_workers 8 \
    --seed 101 \
    --cuda 0 \
    --llm_model gpt-6-luna \
    --llm_base_url https://api.openai.com/v1 \
    --reasoning_effort none \
    --llm_temperature 0.0 \
    --max_api_retries 8 \
    --activity_persona_mode included

python scripts/audit_item_matching.py \
    storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages
