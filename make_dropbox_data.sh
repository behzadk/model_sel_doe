#!/bin/bash
wd="./"
exp_name="spock_manu_stable_1_SMC"

input_dir=$wd"/input_files/input_files_two_species_spock_manu_1"

output_dir=$wd$"output/"$exp_name"/"
echo $output_dir

drop_box_data_dir=$wd"drop_box_results/"$exp_name"_drp_bx/"

# mkdir $drop_box_data_dir


for d in $output_dir* ; do
    echo "$d"
    BASENAME=`basename "$d"`

    # Copy all of experiment analysis folder
    if [ "$BASENAME" == "experiment_analysis" ]; then
    	cp -r $d $drop_box_data_dir

	else
		experiment_folder=$drop_box_data_dir$BASENAME"/"
	    
	    # Make folder for each experiment
		mkdir $experiment_folder

		# Iterate population folders
		for pop in $d"/"* ; do
			pop_basename=`basename "$pop"`
			pop_folder=$experiment_folder$pop_basename"/"

			mkdir $pop_folder
			# Copy model space report
			cp  $pop/"model_space_report.csv" $pop_folder

			# Copy epsilon
			cp  $pop/"epsilon.txt" $pop_folder

			# Copy analysis
			cp -r $pop/"analysis/*" $pop_folder

			echo "$pop"
		done

		# # Copy analysis folder to drpbx
		# cp -r $experiment_folder"/analysis/" $drop_box_data_dir
		#

    fi

done
# cp -r $input_dir $drop_box_data_dir
# cp -r $output_dir $drop_box_data_dir
