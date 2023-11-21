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

# OpenEXR v1.3.9 does not work on Ubuntu 22.04! --> v1.3.8 used

python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
python -m pip install pillow matplotlib tifffile tqdm tensorboard OpenEXR==1.3.8 numpy opencv-python pyqt5 imagecodecs && \
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
echo "You might want to add a bash alias to activate the virtual environment from everywhere:"
echo "echo \"alias mapproc=\\\"source \\\\\\\"$PWD/$virt_env/bin/activate\\\\\\\"\\\"\" >> ~/.bash_aliases"
echo "Then reload your bash aliases via: . ~/.bash_aliases"
