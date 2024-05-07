# create "outputs" folder if it doesn't exist
if [ ! -d "outputs" ]; then
    mkdir "outputs"
    echo "Created 'outputs' folder."
fi

# create "errors" folder if it doesn't exist
if [ ! -d "errors" ]; then
    mkdir "errors"
    echo "Created 'errors' folder."
fi

# iterate through our settings, indexed by this integer
for ((i = 0; i < 3; i++)); do
    for ((j = 0; j < 3; j++)); do
        for ((k = 0; k < 2; k++)); do
            for ((l = 0; l < 10; l++)); do
                for ((m = 0; m < 10; m++)); do

                    # launch our job
                    sbatch saliency_map_generator_main_runscript.sh $i $j $k $l $m
                    sleep 0.5
        
                done
            done
        done
    done
done