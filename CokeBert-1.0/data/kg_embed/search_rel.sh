#!/bin/bash
echo "Argument entity: $1"
echo "=============="
((id=$1-1))
#id=$1
echo $id
echo "=============="
grep -w $id train2id.txt | awk '{print $1,$2,$3}' > del_1
#cat Q_id
echo "=============="
cat del_1 | awk '{print $3}' | sort | uniq > del_2
grep -f del_2 relation2id.txt > del_3
echo "=============="
cat del_1
rm del_1
echo "=============="
cat del_2
rm del_2
echo "=============="
cat del_3
rm del_3

