#!/bin/bash
#SBATCH --job-name=agent4rec-luna-actswap
#SBATCH --account=annavara_668
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=reproduction_outputs/agent4rec-luna-actswap-%j.out
#SBATCH --error=reproduction_outputs/agent4rec-luna-actswap-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RANKINGS_PATH="${RANKINGS_PATH:?Pass the cached full_rankings_1000.npy path with sbatch --export}"
ITEM_RATING_MODE="${ITEM_RATING_MODE:-included}"

case "${ITEM_RATING_MODE}" in
    included)
        DEFAULT_SIMULATION_NAME=agent4rec_lightgcn_luna_1000_5pages_activity_swap
        ;;
    excluded)
        DEFAULT_SIMULATION_NAME=agent4rec_lightgcn_luna_1000_5pages_activity_swap_no_item_rating
        ;;
    *)
        echo "ITEM_RATING_MODE must be included or excluded" >&2
        exit 2
        ;;
esac
SIMULATION_NAME="${SIMULATION_NAME:-${DEFAULT_SIMULATION_NAME}}"
EXPECTED_RANKINGS_SHA256=906e830fc56652cf21a9f2818cd3fa571f9b38d7ff9909f4e11ae72cd15e89d1

cd "${PROJECT_ROOT}"
mkdir -p reproduction_outputs
export PYTHONPATH="${PROJECT_ROOT}/.deps:${PROJECT_ROOT}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-agent4rec"
export PYTHONUNBUFFERED=1

STATUS_FILE="${PROJECT_ROOT}/reproduction_outputs/${SIMULATION_NAME}-${SLURM_JOB_ID:-manual}.status"
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

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not available" >&2
    exit 3
fi
if [[ ! -f "${RANKINGS_PATH}" ]]; then
    echo "Missing cached ranking: ${RANKINGS_PATH}" >&2
    exit 4
fi

verify_sha256() {
    expected=$1
    path=$2
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Unexpected SHA-256 for ${path}: ${actual}" >&2
        exit 5
    fi
}

verify_sha256 "${EXPECTED_RANKINGS_SHA256}" "${RANKINGS_PATH}"
verify_sha256 fbe15e5019b0b05b5cc646859b95515187d3565fcbc9e25c3533f67129662d77 \
    datasets/ml-1m/simulation/all_personas_like_modify.csv
verify_sha256 e133d957d6c30d1e60d489d859089cf26fcd7f422a99b4a9068acfd573be3d8a \
    datasets/ml-1m/simulation/user_statistic.csv
verify_sha256 01c883a0500cf7f2b030350e80a47138c5d4ba5689fe7426099f94cd472a7c5f \
    datasets/ml-1m/simulation/movie_detail.csv
verify_sha256 179b1f7ec8baf96953d22373721383673ef5897c4cde360dcfbf8ad83f946c56 \
    datasets/ml-1m/cf_data/train.txt
verify_sha256 3a57bc6891f7b7137a0b9c11d8ec736d47baf8ea36a0c1f8b332a599149454f8 \
    datasets/ml-1m/cf_data/valid.txt
verify_sha256 9f2c57f8af87139eb7d2a1ee5be33fdb2d58bf646040870042e3c52692b26b56 \
    datasets/ml-1m/cf_data/test.txt

python scripts/check_luna_access.py
python main.py \
    --simulation_name "${SIMULATION_NAME}" \
    --dataset ml-1m \
    --modeltype LightGCN \
    --model_path Saved \
    --rankings_path "${RANKINGS_PATH}" \
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
    --activity_persona_mode included \
    --activity_persona_transform swap_extremes \
    --item_rating_mode "${ITEM_RATING_MODE}"

RUN_DIR="${PROJECT_ROOT}/storage/ml-1m/LightGCN/${SIMULATION_NAME}"
python scripts/audit_item_matching.py "${RUN_DIR}"

if [[ "${ITEM_RATING_MODE}" == excluded ]]; then
    if rg -n "History ratings:" "${RUN_DIR}/running_logs"; then
        echo "A recommended-item historical rating leaked into a rating-hidden run" >&2
        exit 6
    fi
fi
