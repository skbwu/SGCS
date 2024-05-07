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
for ((i = 0; i < 65; i++)); do

    # launch our job
    sbatch cifar10_resnet_apd_main_runscript.sh $i
    sleep 0.5
    
done