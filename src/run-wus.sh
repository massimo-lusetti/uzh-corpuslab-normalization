#!/bin/bash

# train nmt
# uncomment the right setting: wusinit or wusmod
./Main-wus-soft-train.sh

# nmt detailed eval
./Main-wus-sync.sh wusinit wus_phase1/wusinit_sync 5 3 nmt
./Main-wus-sync.sh wusmod wus_phase1/wusmod_sync 5 3 nmt

# sync decoding
./Main-wus-sync.sh wusmod wus_phase1/wusmod_sync 5 3 we
./Main-wus-sync.sh wusmod wus_phase1/wusmod_sync 5 3 ce
./Main-wus-sync.sh wusmod wus_phase1/wusmod_sync 5 3 cw
./Main-wus-sync.sh wusmod wus_phase1/wusmod_sync 5 3 cwe
./Main-wus-sync.sh wusinit wus_phase1/wusinit_sync 5 3 we
./Main-wus-sync.sh wusinit wus_phase1/wusinit_sync 5 3 ce
./Main-wus-sync.sh wusinit wus_phase1/wusinit_sync 5 3 cw
./Main-wus-sync.sh wusinit wus_phase1/wusinit_sync 5 3 cwe

