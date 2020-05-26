#!/bin/bash -e

set -x

while (($#)); do
   case $1 in
     "--config")
       shift
       CONFIG="$1"
       shift
       ;;
     "--data")
       shift
       DATA="$1"
       shift
       ;;
     "-o")
       shift
       OUTPATH="$1"
       shift
       ;;
     "--fixed_model_name")
       shift
       FIXEDMODELNAME="$1"
       shift
       ;;
     "--project")
       shift
       PROJECT="$1"
       shift
       ;;
     *)
       echo "Unknown argument: '$1'"
       exit 1
       ;;
   esac
done

python3 -m rasa_nlu.train --config ${CONFIG} --data ${DATA} -o ${OUTPATH} --fixed_model_name ${FIXEDMODELNAME} --project ${PROJECT}  2>&1 | tee out
cat out
