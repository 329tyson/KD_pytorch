#!/bin/bash

weights=('0.05' '0.1' '0.5' '1.')
convs=('1' '2' '3' '4' '5' '1 2' '1 3' '1 4' '1 5' '2 3' '2 4' '2 5' '3 4' '3 5' '4 5' '1 2 3')

for weight in "${weights[@]}"
do 
    for conv in "${convs[@]}"
    do
        echo executing combination of weight:["${weight}"] convs:["${conv}"]
        python main.py --v --data ../CUB_200_2011/images/ --annotation_train ../labels/label_train_cub200_2011.csv --annotation_val ../labels/label_val_cub200_2011.csv --dataset cub --classes 200 --b 111 --epoch 100 --kd_enabled --pretrain_path ./models/cub_teacher_68.pt --mse_conv "${conv}" --mse_weight "${weight}" --low_ratio 25 --gpu 1
    done
done
