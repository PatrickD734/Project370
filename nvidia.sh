#!/bin/bash
curl -s -X POST "http://localhost:8080/api/tts/GYQ5yGV_-Oc?config=c-fb1074a&alignment=false" &
sleep 15 && docker logs foreign-whispers-api --tail 20 | grep -E "tts|speaker|voice|POST"
