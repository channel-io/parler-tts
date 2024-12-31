ps -ef | grep run_parler_tts_training | grep -v grep | awk '{print $2}' | xargs -r kill -9
