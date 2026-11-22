#!/bin/bash
#SBATCH --job-name=simuser-noact-smoke
#SBATCH --account=annavara_668
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=SimUSER/MovieLens/result/slurm/simuser-noact-smoke-%j.out
#SBATCH --error=SimUSER/MovieLens/result/slurm/simuser-noact-smoke-%j.err

set -euo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.21
source /home1/moonsoo/projects/venv/bin/activate

ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/PQA
AGENT_ROOT=/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
MOVIE_ROOT=${ROOT}/SimUSER/MovieLens
RUN_ROOT=${MOVIE_ROOT}/result/vanilla_common_lightgcn_u1000_no_activity_smoke
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
  --user-ids 0,1,2 --model gpt-6-luna --max-pages 5 --activity-mode removed \
  --workers 3 --user-batch-size 3 --batch-cooldown-seconds 5 --resume

python -B SimUSER/MovieLens/scripts/audit_vanilla_run.py \
  --run-dir "${RUN_ROOT}" \
  --ranking-json "${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.json" \
  --train-split "${CF}/train.txt" --test-split "${CF}/test.txt" --num-users 3

python -B SimUSER/MovieLens/scripts/summarize_vanilla_run.py \
  --run-dir "${RUN_ROOT}" --num-users 3
