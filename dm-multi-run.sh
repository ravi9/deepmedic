
#!/bin/bash


CSV_FILE=/Share/ravi/upenn/data/brainmage-tcga-test-ds/tcga-test-ds-pre.csv
#CSV_FILE=/Share/ravi/upenn/data/brainmage-tcga-test-ds/tcga-test-ds-pre-5rows.csv
CSV_FILE=/Share/ravi/upenn/data/brainmage-tcga-test-ds/tcga-test-ds-pre-1rows.csv
MOPREFIX=/home/rpanchum/upenn/ravi9-dm-tf2/examples/tcga/saved_models/skullStripping_modalityAgnostic/

i=0
NUM_ROWS=$(wc -l $CSV_FILE)

while IFS=, read -r sub_id input_img mask_img input_img_pre
do
    echo "Starting $sub_id"

    echo "$mask_img" > $MOPREFIX/testGtLabels.cfg
    echo "$input_img_pre" > $MOPREFIX/testChannels_t1c.cfg

    ./deepMedicRun -model $MOPREFIX/modelConfig.txt \
               -test $MOPREFIX/testConfig.cfg \
               -load /home/rpanchum/upenn/ravi9-dm-tf2/examples/tcga/savedmodels/deepmedic-4-ov.model.ckpt

   i=$((i+1))
   echo "##########^^^^^^^^^^^^^^^^^###############"
   echo "Done $i / $NUM_ROWS : $sub_id"
   echo "##########^^^^^^^^^^^^^^^^^###############"

done < $CSV_FILE
