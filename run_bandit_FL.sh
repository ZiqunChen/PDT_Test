# Federated Learning
# Collect data
python3 collect_data.py --env bandit --envs 30000 --H 100 --dim 3 --var 0.3 --cov 0.0 --envs_eval 200
# Train
python3 FL.py --env bandit --envs 30000 --H 100 --dim 3 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --seed 1
# Evaluate, choose an appropriate epoch
python3 FL_eval.py --env bandit --envs 30000 --H 100 --dim 3 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch 400 --n_eval 200 --seed 1
