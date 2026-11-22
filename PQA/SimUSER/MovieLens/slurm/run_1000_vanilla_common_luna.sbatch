#!/bin/bash
#SBATCH --job-name=simuser-vanilla-1000
#SBATCH --account=annavara_668
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=SimUSER/MovieLens/result/slurm/simuser-vanilla-1000-%j.out
#SBATCH --error=SimUSER/MovieLens/result/slurm/simuser-vanilla-1000-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/PQA
AGENT_ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
MOVIE_ROOT=${ROOT}/SimUSER/MovieLens
PROFILE_ROOT=${MOVIE_ROOT}/result/vanilla_profiles_u1000
MEMORY_ROOT=${MOVIE_ROOT}/result/vanilla_memory_u1000
RUN_ROOT=${MOVIE_ROOT}/result/vanilla_common_lightgcn_u1000
RANKING_JSON=${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.json
RANKING_MANIFEST=${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.manifest.json
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
  --stage all --num-users 1000 --model gpt-6-luna \
  --workers 4 --user-batch-size 4 --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/prepare_vanilla_memory.py \
  --raw-dir "${RAW}" --cf-data-dir "${CF}" --simulation-data-dir "${SIM}" \
  --output-dir "${MEMORY_ROOT}" --env-file "${ENV_FILE}" \
  --stage all --num-users 1000 --workers 4 --user-batch-size 4 \
  --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/run_vanilla_common.py \
  --ranking-json "${RANKING_JSON}" --ranking-manifest "${RANKING_MANIFEST}" \
  --profiles "${PROFILE_ROOT}/user_profiles.json" \
  --episodic-dir "${MEMORY_ROOT}/episodic" --kg "${MEMORY_ROOT}/kg_memory.json" \
  --movie-detail "${SIM}/movie_detail.csv" \
  --train-split "${CF}/train.txt" --valid-split "${CF}/valid.txt" --test-split "${CF}/test.txt" \
  --result-dir "${RUN_ROOT}" --env-file "${ENV_FILE}" \
  --num-users 1000 --model gpt-6-luna --max-pages 5 \
  --workers 4 --user-batch-size 4 --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/audit_vanilla_run.py \
  --run-dir "${RUN_ROOT}" --ranking-json "${RANKING_JSON}" \
  --train-split "${CF}/train.txt" --test-split "${CF}/test.txt" --num-users 1000

python -B SimUSER/MovieLens/scripts/summarize_vanilla_run.py \
  --run-dir "${RUN_ROOT}" --num-users 1000

python -B SimUSER/MovieLens/scripts/analyze_activity_degree.py \
  --run-dir "${RUN_ROOT}" --dataset-dir "${AGENT_ROOT}/datasets/ml-1m" \
  --output "${MOVIE_ROOT}/analysis/POPULARITY_BIAS_BY_ACTIVITY_U1000.md" \
  --num-users 1000
