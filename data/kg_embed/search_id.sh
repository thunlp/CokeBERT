#!/bin/bash
echo "Argument entity: $1"
echo "=============="
((id=$1-1))
#id=$1
echo $id
echo "=============="
grep -w $id entity2id.txt | awk '{print $1}' > Q_id
#cat Q_id
echo "=============="
grep -f Q_id entity_map.txt
rm Q_id
