#!/bin/bash

export FREESURFER_HOME=/Applications/freesurfer_dev
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4

# for SUBI in 017 020 021 023 024 025 026 
for SUBI in 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 120 121 122 123 124 125
do
    segmentHA_T1.sh "Sub${SUBI}/Sub${SUBI}fs"
done