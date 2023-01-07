To obtain the modified model, run the following command under the Socrates - 1.4.0 folder

time python source/run_causal.py --spec benchmark/causal/credit/spec_age_processed.json --algorithm causal --dataset credit
time python source/run_causal.py --spec benchmark/causal/credit/spec_gender_processed.json --algorithm causal --dataset credit

time python source/run_causal.py --spec benchmark/causal/bank/spec_age_processed.json --algorithm causal --dataset bank

time python source/run_causal.py --spec benchmark/causal/census/spec_race_processed.json --algorithm causal --dataset census
time python source/run_causal.py --spec benchmark/causal/census/spec_gender_processed.json --algorithm causal --dataset census

time python source/run_causal.py --spec benchmark/causal/compas/spec_race_processed.json --algorithm causal --dataset compas
time python source/run_causal.py --spec benchmark/causal/compas/spec_gender_processed.json --algorithm causal --dataset compas