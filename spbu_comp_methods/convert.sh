#!/bin/bash

root="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}"   )" &> /dev/null && pwd   )"

echo $root

for dir in $(find . -maxdepth 1 -type d | grep -v .git | grep -vE '^\.$'); do 
  cd $dir
  #jupyter nbconvert --to script *.ipynb
  for py in $(find . -name '*.py'); do
    i=$(grep -nE '(def|=|\(\))' $py | head -n 1 | cut -d : -f 1)
    head -n $i $py | grep import | tee $root/module.py
code=<<EOF
  def $dir():
EOF
  done
  cd ..
done
