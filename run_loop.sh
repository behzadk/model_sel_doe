which python3

 for i in {0..100..1}
    do
       echo "Welcome $i times"
       # python3 run_boost_rpr.py --config exp_configs/experiment_config_BK_manu_two_species_5.yaml --exp_suffix $i >> log.out_$i
       python3 run_boost_rpr.py --config exp_configs/experiment_config_BK_manu_three_species_stable_7.yaml --exp_suffix $i >> log.out_$i
 done
