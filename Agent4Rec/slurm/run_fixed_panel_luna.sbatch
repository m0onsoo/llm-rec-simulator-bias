#!/bin/bash
#SBATCH --job-name=agent4rec-luna-panel
#SBATCH --account=annavara_668
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=reproduction_outputs/agent4rec-luna-panel-%j.out
#SBATCH --error=reproduction_outputs/agent4rec-luna-panel-%j.err

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

# An export made in a separate interactive shell is not inherited by Codex's
# sbatch process. Load the local, Git-ignored environment file when present.
if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not available in the SLURM environment" >&2
    exit 2
fi

python scripts/check_luna_access.py

REPEATS=${REPEATS:-2}
SIM_PREFIX=${SIM_PREFIX:-fixed_panel_luna}
for repeat in $(seq 1 "${REPEATS}"); do
    python main.py \
        --simulation_name "${SIM_PREFIX}_repeat_${repeat}" \
        --dataset ml-1m \
        --modeltype LightGCN \
        --model_path Saved \
        --n_avatars 3 \
        --items_per_page 4 \
        --max_pages 1 \
        --execution_mode parallel \
        --seed 101 \
        --cuda 0 \
        --llm_model gpt-6-luna \
        --llm_base_url https://api.openai.com/v1 \
        --reasoning_effort none \
        --llm_temperature 0.0 \
        --max_api_retries 8 \
        --max_workers 3
done
