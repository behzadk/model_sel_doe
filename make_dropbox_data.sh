#!/bin/bash
exp_name="two_species_stable_12"
wd="./"
output_dir=$wd$"output/"$exp_name"/Population_0/"
input_dir=$wd"input_files_two_species_0"
echo $output_dir

drop_box_data_dir=$output_dir$exp_name"_drp_bx/"

mkdir $drop_box_data_dir

cp -r $input_dir $drop_box_data_dir
cp -r $output_dir"analysis" $drop_box_data_dir
cp -r $output_dir"analysis/KS_dist_plots" $drop_box_data_dir
cp -r $output_dir"analysis/model_order.txt" $drop_box_data_dir
cp -r $output_dir"/model_sim_params" $drop_box_data_dir