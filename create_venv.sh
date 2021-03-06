#!/bin/bash

abortOnFailure() {
    status=$?
    if [ "$status" -ne 0 ];
    then
        echo "ERROR: $status" 1>&2;
        exit $status
    fi  
}

virt_env=venv_map_processing

if [ -z "$(find /usr/lib -iname "libIex*.so" 2> /dev/null)" ];
then
    echo "Please install libilmbase." 1>&2
    exit 1
fi

if ! [ -d "$virt_env" ];
then
    python3 -m venv "$virt_env"
    abortOnFailure
fi

sed -i "s#PATH=\"\$VIRTUAL_ENV#PATH=\"$PWD:\$VIRTUAL_ENV#g" $virt_env/bin/activate
source "$virt_env/bin/activate"

python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
python -m pip install pillow matplotlib tifffile tqdm tensorboard openexr numpy opencv-python && \
abortOnFailure

python -m pip install open3d
if [ "$?" -ne "0" ];
then
    echo "The virtual environment ${virt_env} was created successfully." 1>&2
    echo "However, no Open3D verison was found for $(python --version)." 1>&2
    echo "Please install Open3D in the virtual environment ${virt_env} manually." 1>&2
    echo "If you build Open3D from source, do not forget to activate your virtual environment before." 1>&2
fi

echo "SUCCESS"
