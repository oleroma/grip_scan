#!/bin/bash
# run_hand_variant.sh SEED: random hand -> curl it round GripV -> solve the grip rig for it -> Blender check.
# Logs the time of every step to $W/timing.txt. Needs: HSR (hand_scan_rig + gripscan checkout), PY (bpy venv),
# BASE (a big_hand work dir with hold_rig.blend, web_landmarks.json, web_skin.npz, grip_sdf.npz),
# RIG (grip rig work dir: rig_state.npz rig.json bones.npz Ax.npy grip_rig.blend).
set -e
SEED=$1
HSR=${HSR:-/home/claude/hsr}; PY=${PY:-/home/claude/venv/bin/python}
BASE=${BASE:-/tmp/claude-0/hs/work/big_hand}; RIG=${RIG:-/tmp/claude-0/hs/work/grip_rig}
HERE=$(cd "$(dirname "$0")" && pwd)
W=/tmp/claude-0/hs/work/rh_$SEED; G=/tmp/claude-0/hs/work/grip_rh_$SEED
mkdir -p $W $G
cp $BASE/{hold_rig.blend,web_landmarks.json,web_skin.npz,grip_sdf.npz} $W/
cp $RIG/{rig_state.npz,rig.json,bones.npz,Ax.npy,grip_rig.blend} $G/
T=$W/timing.txt; : > $T
step() { local name=$1; shift; local t0=$(date +%s); "$@" > $W/log_$name.txt 2>&1; echo "$name $(( $(date +%s) - t0 )) s" | tee -a $T; }
export BIG_HAND_WORK=$W PYTHONPATH=$HSR
cd $HSR/hand_scan_rig
step random_hand   $PY big_hand.py random --seed $SEED
step curl_start    $PY big_hand.py start
step curl_solve    $PY big_hand.py solve --nfev 60
step curl_blend    $PY big_hand.py blend
cd $HERE
step hand_extract  $PY extract_hand.py --hold $W/hand_on_grip_big.blend --work $G
step grip_solve    $PY -u solve_grip.py --work $G --smooth 5
step grip_blender  $PY -u pose_in_blender.py --rig $G/grip_rig.blend --work $G --out $G/grip_rig_posed.blend
echo "done seed $SEED" | tee -a $T
