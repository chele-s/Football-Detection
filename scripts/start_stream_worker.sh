#!/bin/bash

INPUT_URL=$1
OUTPUT_URL=$2

if [ -z "$INPUT_URL" ]; then
    echo "Uso: ./start_stream_worker.sh <INPUT_URL> [OUTPUT_URL]"
    echo ""
    echo "Ejemplos:"
    echo "  ./start_stream_worker.sh rtmp://source.com/live/input rtmp://output.com/live/output"
    echo "  ./start_stream_worker.sh https://youtube.com/watch?v=VIDEO_ID"
    exit 1
fi

echo "===== Football Tracker Stream Worker ====="
echo "Input: $INPUT_URL"
echo "Output: ${OUTPUT_URL:-DEBUG MODE}"
echo ""

if [ -z "$OUTPUT_URL" ]; then
    python main.py stream --input "$INPUT_URL" --debug
else
    python main.py stream --input "$INPUT_URL" --output "$OUTPUT_URL"
fi
