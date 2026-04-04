#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/garycolonna

while true; do
    echo "[$(date)] Starting server_receiver.py..." >> /tmp/hub.log
    python3 -u server_receiver.py >> /tmp/hub.log 2>&1
    echo "[$(date)] Server died — restarting in 3s..." >> /tmp/hub.log
    sleep 3
done
