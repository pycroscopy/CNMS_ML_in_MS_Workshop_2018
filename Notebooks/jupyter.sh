#!/bin/bash

# running jupyter notebooks on NFS occasionally has issues
# this should fix any NFS related problem

export TDIR=$(mktemp -d -p /local/tmp)
export IPYTHONDIR=${TDIR}/ipyd
export JUPYTER_DATA_DIR=${TDIR}/jdd
unset XDG_RUNTIME_DIR

jupyter notebook
