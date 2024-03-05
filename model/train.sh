nohup python3 custom_train.py
echo "Training done" | mail -s "Training done" "ADDRESS@ncsu.edu" -A "nohup.out"