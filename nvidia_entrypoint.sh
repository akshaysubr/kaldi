#!/bin/bash
set -e
cat <<EOF

===========
== Kaldi ==
===========

NVIDIA Release ${NVIDIA_KALDI_VERSION} (build ${NVIDIA_BUILD_ID})

Container image Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
Copyright 2017, Kaldi contributors.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.
EOF

if [[ "$(find /usr -name libcuda.so.1) " == " " || "$(ls /dev/nvidiactl) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use 'nvidia-docker run' to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker ."
fi

#if [[ "$(df -k /dev/shm |grep ^shm |awk '{print $2}') " == "65536 " ]]; then
#  echo
#  echo "NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be"
#  echo "   insufficient for Kaldi.  NVIDIA recommends the use of the following flags:"
#  echo "   nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ..."
#fi

echo

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
