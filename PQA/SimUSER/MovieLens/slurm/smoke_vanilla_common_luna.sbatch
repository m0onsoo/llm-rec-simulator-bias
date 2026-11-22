#!/bin/bash
#SBATCH --job-name=simuser-vanilla-smoke
#SBATCH --account=annavara_668
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=SimUSER/MovieLens/result/slurm/smoke-vanilla-common-%j.out
#SBATCH --error=SimUSER/MovieLens/result/slurm/smoke-vanilla-common-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/PQA
AGENT_ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
MOVIE_ROOT=${ROOT}/SimUSER/MovieLens
PROFILE_ROOT=${MOVIE_ROOT}/result/vanilla_profiles_smoke
MEMORY_ROOT=${MOVIE_ROOT}/result/vanilla_memory_smoke
RUN_ROOT=${MOVIE_ROOT}/result/vanilla_common_smoke
RANKING_JSON=${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u200_top20.json
RANKING_MANIFEST=${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u200_top20.manifest.json
RAW=${AGENT_ROOT}/datasets/ml-1m/raw_data
CF=${AGENT_ROOT}/datasets/ml-1m/cf_data
SIM=${AGENT_ROOT}/datasets/ml-1m/simulation
ENV_FILE=${AGENT_ROOT}/.env

cd "${ROOT}"
export PYTHONUNBUFFERED=1
mkdir -p "${MOVIE_ROOT}/result/slurm"

python -B SimUSER/MovieLens/scripts/prepare_vanilla_profiles.py \
  --raw-dir "${RAW}" --cf-data-dir "${CF}" --simulation-data-dir "${SIM}" \
  --output-dir "${PROFILE_ROOT}" --env-file "${ENV_FILE}" \
  --stage all --user-ids 0,1,2 --model gpt-6-luna \
  --workers 3 --user-batch-size 3 --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/prepare_vanilla_memory.py \
  --raw-dir "${RAW}" --cf-data-dir "${CF}" --simulation-data-dir "${SIM}" \
  --output-dir "${MEMORY_ROOT}" --env-file "${ENV_FILE}" \
  --stage all --user-ids 0,1,2 --workers 3 --user-batch-size 3 --resume

python -B SimUSER/MovieLens/scripts/run_vanilla_common.py \
  --ranking-json "${RANKING_JSON}" --ranking-manifest "${RANKING_MANIFEST}" \
  --profiles "${PROFILE_ROOT}/user_profiles.json" \
  --episodic-dir "${MEMORY_ROOT}/episodic" --kg "${MEMORY_ROOT}/kg_memory.json" \
  --movie-detail "${SIM}/movie_detail.csv" \
  --train-split "${CF}/train.txt" --valid-split "${CF}/valid.txt" --test-split "${CF}/test.txt" \
  --result-dir "${RUN_ROOT}" --env-file "${ENV_FILE}" \
  --user-ids 0,1,2 --model gpt-6-luna --max-pages 5 \
  --workers 3 --user-batch-size 3 --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/audit_vanilla_run.py \
  --run-dir "${RUN_ROOT}" --ranking-json "${RANKING_JSON}" \
  --train-split "${CF}/train.txt" --test-split "${CF}/test.txt" --num-users 3
