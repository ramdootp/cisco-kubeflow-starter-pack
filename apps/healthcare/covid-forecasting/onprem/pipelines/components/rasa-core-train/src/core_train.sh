#!/bin/bash -e
  
set -x

while (($#)); do
   case $1 in
     "--domain")
       shift
       DOMAIN="$1"
       shift
       ;;
     "--stories")
       shift
       STORIES="$1"
       shift
       ;;
     "--out")
       shift
       MODEL="$1"
       shift
       ;;
     *)
       echo "Unknown argument: '$1'"
       exit 1
       ;;
   esac
done

pwd
ls -l
ls -l /ml
python3 -m rasa_core.train -d ${DOMAIN} -s ${STORIES} -o ${MODEL} 2>&1 | tee out
accuracy=`grep acc out | tail -1 | awk '{print $(NF)}'`
echo "{\"metrics\": [{\"name\": \"core-accuracy\", \"numberValue\": ${accuracy}, \"format\": \"PERCENTAGE\"}]}" > /mlpipeline-metrics.json
