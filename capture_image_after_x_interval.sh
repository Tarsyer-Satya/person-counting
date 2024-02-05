#!/bin/bash

# Set the time interval in seconds
time_interval=20

while true; do
    # Your commands go here
    echo "Running your commands at $(date)"
    ffmpeg -hide_banner -loglevel error -rtsp_transport tcp -ss 00:00:03 -i "rtsp://admin:Admin@123@10.3.247.50:554/Streaming/Channels/301" -frames:v 1 -q:v 2 /tmp/image.jpg -y
    # Sleep for the specified time interval
    sleep $time_interval
done

