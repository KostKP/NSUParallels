#!/bin/bash

folder="./out"

cd "$folder" || exit

for bin_file in *; do
    ./"$bin_file"
    echo ""
done

echo "Finished running all files."
