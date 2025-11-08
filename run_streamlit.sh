#!/bin/bash

echo "Starting Football Detection Streamlit App..."
echo "=============================================="
echo ""

if command -v streamlit &> /dev/null; then
    echo "✓ Streamlit found"
else
    echo "✗ Streamlit not found. Installing..."
    pip install streamlit --quiet
fi

echo ""
echo "Starting ngrok tunnel..."

if command -v ngrok &> /dev/null; then
    ngrok http 8501 > /dev/null 2>&1 &
    NGROK_PID=$!
    sleep 3
    
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
    
    if [ ! -z "$NGROK_URL" ]; then
        echo "✓ ngrok tunnel active"
        echo "Public URL: $NGROK_URL"
    else
        echo "⚠ ngrok tunnel not available"
    fi
else
    echo "⚠ ngrok not found - app will run on localhost:8501 only"
fi

echo ""
echo "=============================================="
echo "Starting Streamlit on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
