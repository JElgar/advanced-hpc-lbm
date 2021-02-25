squeue -u za18968 -h | awk '{ print $1 }' | xargs scancel
