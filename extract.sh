#!/usr/bin/env sh
[ ! $# -eq 1 ] && echo "Usage: $0 filename" && exit 1
IN=$(cat "$1" | head -n 1)
(dd if=/dev/zero of="$1_cols.txt" bs=1 count=0 2> /dev/null)
IFS=","
for i in $IN
do
    echo "$i" >> "$1_cols.txt"
done