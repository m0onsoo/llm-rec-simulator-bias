#!/bin/bash
#SBATCH --job-name=agent4rec-luna-1000-noact
#SBATCH --account=annavara_668
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=reproduction_outputs/agent4rec-luna-1000-noact-%j.out
#SBATCH --error=reproduction_outputs/agent4rec-luna-1000-noact-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
WITH_ACTIVITY_RUN=${ROOT}/storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_batch4_from200
RANKINGS=${WITH_ACTIVITY_RUN}/rankings/full_rankings_1000.npy
cd "${ROOT}"
export PYTHONPATH="${ROOT}/.deps:${ROOT}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-agent4rec"
export PYTHONUNBUFFERED=1
STATUS_FILE="${ROOT}/reproduction_outputs/agent4rec-luna-1000-noact-${SLURM_JOB_ID:-manual}.status"
termination_signal=none
record_exit_status() {
    exit_code=$?
    printf 'job_id=%s\nexit_code=%s\nsignal=%s\nfinished_at=%s\n' \
        "${SLURM_JOB_ID:-manual}" "${exit_code}" "${termination_signal}" \
        "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
}
handle_term() { termination_signal=TERM; exit 143; }
handle_int() { termination_signal=INT; exit 130; }
trap record_exit_status EXIT
trap handle_term TERM
trap handle_int INT

if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not available" >&2
    exit 2
fi
if [[ ! -f "${RANKINGS}" ]]; then
    echo "Complete the with-activity run first; missing ${RANKINGS}" >&2
    exit 3
fi

python scripts/check_luna_access.py
python main.py \
    --simulation_name agent4rec_lightgcn_luna_1000_5pages_no_activity \
    --dataset ml-1m \
    --modeltype LightGCN \
    --model_path Saved \
    --rankings_path "${RANKINGS}" \
    --resume \
    --n_avatars 1000 \
    --items_per_page 4 \
    --max_pages 5 \
    --execution_mode parallel \
    --max_workers 2 \
    --avatar_batch_size 2 \
    --batch_pause_seconds 10 \
    --seed 101 \
    --cuda 0 \
    --llm_model gpt-6-luna \
    --llm_base_url https://api.openai.com/v1 \
    --reasoning_effort none \
    --llm_temperature 0.0 \
    --max_api_retries 8 \
    --activity_persona_mode excluded

python scripts/audit_item_matching.py \
    storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_no_activity

python scripts/update_activity_bias_report.py \
    --with-activity "${WITH_ACTIVITY_RUN}" \
    --without-activity \
        "${ROOT}/storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_no_activity"
