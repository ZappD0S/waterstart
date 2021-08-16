#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

OPTIONS=
LONGOPTS=with-cuda,without-cuda,conda-path:

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
eval set -- "$PARSED"

cuda=false
condaPath=

while true; do
    case "$1" in
    --with-cuda)
        cuda=true
        shift
        ;;
    --without-cuda)
        cuda=false
        shift
        ;;
    --conda-path)
        condaPath="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "programming error"
        exit 1
        ;;
    esac
done

if [ $condaPath ]; then
    PATH="$condaPath:$PATH"
fi

source activate

create_env=(conda create -y -n waterstart --file conda_requirements.txt)

if [ $cuda = false ]; then
    create_env+=(cpuonly)
fi

create_env+=(-c pytorch -c conda-forge)

"${create_env[@]}"
