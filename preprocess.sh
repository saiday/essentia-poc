#!/usr/bin/env bash
set -e
ffmpeg -y -i "$1" -ac 1 -ar 16000 -sample_fmt flt -c:a pcm_f32le input_16k_mono.wav
