#!/bin/bash -e
  
set -x

while (($#)); do
   case $1 in
     "--core")
       shift
       DIALOGUE_MODEL="$1"
       shift
       ;;
     "--nlu")
       shift
       NLU_MODEL="$1"
       shift
       ;;
     "--stories")
       shift
       STORIES="$1"
       shift
       ;;
     "-o")
       shift
       RESULTS="$1"
       shift
       ;;
     *)
       echo "Unknown argument: '$1'"
       exit 1
       ;;
   esac
done

python3 -m rasa_core.evaluate default --e2e --core ${DIALOGUE_MODEL} --nlu ${NLU_MODEL} --stories ${STORIES} -o ${RESULTS} 2>&1 | tee out
f1_score=`grep F1-Score out | tail -1 | awk '{print $(NF)}'`
precision=`grep Precision out | tail -1 | awk '{print $(NF)}'`
echo "{\"metrics\": [{\"name\": \"f1-score\", \"numberValue\": ${f1_score}, \"format\": \"PERCENTAGE\"}]}" > /mlpipeline-metrics.json
