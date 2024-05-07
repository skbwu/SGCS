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

# create "models" folder if it doesn't exist
if [ ! -d "models" ]; then
    mkdir "models"
    echo "Created 'models' folder."
fi

# iterate through our settings, indexed by this integer
for ((i = 0; i < 2; i++)); do
    for ((j = 0; j < 3; j++)); do
        for ((k = 0; k < 50; k++)); do

            # launch our job
            sbatch mnist_family_resnet_main_runscript.sh $i $j $k
            sleep 0.5
        
        done
    done
done