#!/bin/bash
# Run continued-training arms with at most $MAXJ concurrent jobs.
# usage: MAXJ=2 bash vision/run_arms.sh <session> <arm> [<session> <arm> ...]
#   arm none|mimetic|random0|agent  ->  resumes <session>/<ckpt> into <session>/arm_<arm>
cd <project>
PY=.venv/Scripts/python.exe; MAXJ=${MAXJ:-2}
while [ $# -gt 1 ]; do
  S=$1; A=$2; shift 2
  case $A in agent) C=edited;; *) C=$A;; esac
  CK=$(ls $S/${C}_step_*.pt | head -1)
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n; done
  echo "$(date +%T) start $S arm_$A from $CK"
  $PY vision/train.py --init $CK --out $S/arm_$A --compile > $S/arm_$A.log 2>&1 &
done
wait
echo "$(date +%T) all done"
