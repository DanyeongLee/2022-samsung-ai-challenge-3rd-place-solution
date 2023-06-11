for config in gem1 gem1-gamma11 gem1-gamma12 gem1-gamma11-gc2
do
    for seed in 0 1 2
    do
        for fold in 0 1 2 3 4 5 6 7 8 9
        do
            python train.py configs/${config}.yaml --seed ${seed} --fold ${fold}
        done
    done
done