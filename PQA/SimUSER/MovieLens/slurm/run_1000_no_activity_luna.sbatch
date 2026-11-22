#!/bin/bash
#SBATCH --job-name=simuser-noact-1000
#SBATCH --account=annavara_668
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=SimUSER/MovieLens/result/slurm/simuser-noact-1000-%j.out
#SBATCH --error=SimUSER/MovieLens/result/slurm/simuser-noact-1000-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/PQA
AGENT_ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
MOVIE_ROOT=${ROOT}/SimUSER/MovieLens
RUN_ROOT=${MOVIE_ROOT}/result/vanilla_common_lightgcn_u1000_no_activity
WITH_ACTIVITY_ROOT=${MOVIE_ROOT}/result/vanilla_common_lightgcn_u1000
CF=${AGENT_ROOT}/datasets/ml-1m/cf_data
SIM=${AGENT_ROOT}/datasets/ml-1m/simulation

cd "${ROOT}"
export PYTHONUNBUFFERED=1
mkdir -p "${MOVIE_ROOT}/result/slurm"

python -B SimUSER/MovieLens/scripts/run_vanilla_common.py \
  --ranking-json "${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.json" \
  --ranking-manifest "${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.manifest.json" \
  --profiles "${MOVIE_ROOT}/result/vanilla_profiles_u1000/user_profiles.json" \
  --episodic-dir "${MOVIE_ROOT}/result/vanilla_memory_u1000/episodic" \
  --kg "${MOVIE_ROOT}/result/vanilla_memory_u1000/kg_memory.json" \
  --movie-detail "${SIM}/movie_detail.csv" \
  --train-split "${CF}/train.txt" --valid-split "${CF}/valid.txt" --test-split "${CF}/test.txt" \
  --result-dir "${RUN_ROOT}" --env-file "${AGENT_ROOT}/.env" \
  --num-users 1000 --model gpt-6-luna --max-pages 5 --activity-mode removed \
  --workers 4 --user-batch-size 4 --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/audit_vanilla_run.py \
  --run-dir "${RUN_ROOT}" \
  --ranking-json "${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.json" \
  --train-split "${CF}/train.txt" --test-split "${CF}/test.txt" --num-users 1000

python -B SimUSER/MovieLens/scripts/summarize_vanilla_run.py \
  --run-dir "${RUN_ROOT}" --num-users 1000

python -B SimUSER/MovieLens/scripts/analyze_activity_degree.py \
  --run-dir "${RUN_ROOT}" --dataset-dir "${AGENT_ROOT}/datasets/ml-1m" \
  --output "${MOVIE_ROOT}/analysis/POPULARITY_BIAS_BY_ACTIVITY_U1000_NO_ACTIVITY.md" \
  --num-users 1000 --activity-prompt removed --artifact-tag simuser_no_activity

python -B SimUSER/MovieLens/scripts/compare_activity_ablation.py \
  --with-activity-run "${WITH_ACTIVITY_ROOT}" --without-activity-run "${RUN_ROOT}" \
  --output "${MOVIE_ROOT}/analysis/ACTIVITY_PROMPT_ABLATION_U1000.md" --num-users 1000
