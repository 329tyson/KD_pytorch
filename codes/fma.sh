#!/bin/bash

cd logs
ag MAX_ACC > ../maxlog.txt
cd ..

ret=0
cut -f 5 -d : maxlog.txt | while read line; do
    if [ $(echo "$line>$ret"|bc) -eq 1 ]; then
        ret=$line
    fi
done
cd logs
echo $(ag "MAX_ACCURACY : $ret")
echo successive experiments : $(ag MAX| wc -l)
echo failed     experiments : $(ag fail| wc -l)
cd ..

rm maxlog.txt
