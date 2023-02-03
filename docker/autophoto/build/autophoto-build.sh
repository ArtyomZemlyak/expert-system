#!/usr/bin/env bash

# Update conda to newest version:
conda update -y --name base --channel defaults conda  && \

# Checking if CPU supported AVX:
AVX="$(grep -o 'avx[^ ]*' /proc/cpuinfo)"
if [[ $AVX == *"avx"* ]]
then
    echo "AVX supported CPU founded!"
    AVX=true
else
    echo "AVX supported CPU could not be found!"
    AVX=false
fi

# Checking if GPU supported CUDA:
if ! command -v lspci &> /dev/null
then
    echo "Package lspci could not be found!"
    echo "Installing lspci..."
    if command -v apt &> /dev/null
    then
        apt install -y pciutils
    elif command -v yum &> /dev/null
    then
        yum install -y pciutils
    fi
    if ! command -v lspci &> /dev/null
    then
        echo "Cant install lspci!"
        GPU=""
    else
        echo "Successfully installed lspci!"
        GPU="$(lspci | grep -i nvidia)"
    fi
else
    echo "lspci founded!"
    GPU="$(lspci | grep -i nvidia)"
fi
if [[ $GPU == *"GeForce RTX 3090"* ]] || \
[[ $GPU == *"GeForce RTX 3080"* ]] || \
[[ $GPU == *"GeForce RTX 3070"* ]] || \
[[ $GPU == *"NVIDIA TITAN RTX"* ]] || \
[[ $GPU == *"Geforce RTX 2080 Ti"* ]] || \
[[ $GPU == *"Geforce RTX 2070"* ]] || \
[[ $GPU == *"GeForce RTX 2060"* ]] || \
[[ $GPU == *"NVIDIA TITAN V"* ]] || \
[[ $GPU == *"NVIDIA TITAN Xp"* ]] || \
[[ $GPU == *"GeForce GTX 1080 Ti"* ]] || \
[[ $GPU == *"NVIDIA TITAN X"* ]] || \
[[ $GPU == *"GeForce GTX 1080"* ]] || \
[[ $GPU == *"GeForce GTX 1070 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 1070"* ]] || \
[[ $GPU == *"GeForce GTX 1050"* ]] || \
[[ $GPU == *"GeForce GTX TITAN X"* ]] || \
[[ $GPU == *"GeForce GTX TITAN Z"* ]] || \
[[ $GPU == *"GeForce GTX TITAN Black"* ]] || \
[[ $GPU == *"GeForce GTX TITAN"* ]] || \
[[ $GPU == *"GeForce GTX 980 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 980"* ]] || \
[[ $GPU == *"GeForce GTX 970"* ]] || \
[[ $GPU == *"GeForce GTX 960"* ]] || \
[[ $GPU == *"GeForce GTX 950"* ]] || \
[[ $GPU == *"GeForce GTX 780 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 780"* ]] || \
[[ $GPU == *"GeForce GTX 770"* ]] || \
[[ $GPU == *"GeForce GTX 760"* ]] || \
[[ $GPU == *"GeForce GTX 750 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 750"* ]] || \
[[ $GPU == *"GeForce GTX 690"* ]] || \
[[ $GPU == *"GeForce GTX 680"* ]] || \
[[ $GPU == *"GeForce GTX 670"* ]] || \
[[ $GPU == *"GeForce GTX 660 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 660"* ]] || \
[[ $GPU == *"GeForce GTX 650 Ti BOOST"* ]] || \
[[ $GPU == *"GeForce GTX 650 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 650"* ]] || \
[[ $GPU == *"GeForce GTX 560 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 550 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 460"* ]] || \
[[ $GPU == *"GeForce GTS 450"* ]] || \
[[ $GPU == *"GeForce GTX 590"* ]] || \
[[ $GPU == *"GeForce GTX 580"* ]] || \
[[ $GPU == *"GeForce GTX 570"* ]] || \
[[ $GPU == *"GeForce GTX 480"* ]] || \
[[ $GPU == *"GeForce GTX 470"* ]] || \
[[ $GPU == *"GeForce GTX 465"* ]] || \
[[ $GPU == *"GeForce GT 740"* ]] || \
[[ $GPU == *"GeForce GT 730"* ]] || \
[[ $GPU == *"GeForce GT 720"* ]] || \
[[ $GPU == *"GeForce GT 705"* ]] || \
[[ $GPU == *"GeForce GT 640"* ]] || \
[[ $GPU == *"GeForce GT 630"* ]] || \
[[ $GPU == *"GeForce GT 620"* ]] || \
[[ $GPU == *"GeForce GT 610"* ]] || \
[[ $GPU == *"GeForce GT 520"* ]] || \
[[ $GPU == *"GeForce GT 440"* ]] || \
[[ $GPU == *"GeForce GT 440"* ]] || \
[[ $GPU == *"GeForce GT 430"* ]] || \
[[ $GPU == *"Geforce RTX 3060"* ]] || \
[[ $GPU == *"Geforce RTX 3060Ti"* ]] || \
[[ $GPU == *"Geforce RTX 3060 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 980M"* ]] || \
[[ $GPU == *"GeForce GTX 970M"* ]] || \
[[ $GPU == *"GeForce GTX 965M"* ]] || \
[[ $GPU == *"GeForce GTX 960M"* ]] || \
[[ $GPU == *"GeForce GTX 950M"* ]] || \
[[ $GPU == *"GeForce 940M"* ]] || \
[[ $GPU == *"GeForce 930M"* ]] || \
[[ $GPU == *"GeForce 920M"* ]] || \
[[ $GPU == *"GeForce 910M"* ]] || \
[[ $GPU == *"GeForce GTX 880M"* ]] || \
[[ $GPU == *"GeForce GTX 870M"* ]] || \
[[ $GPU == *"GeForce GTX 860M"* ]] || \
[[ $GPU == *"GeForce GTX 850M"* ]] || \
[[ $GPU == *"GeForce 840M"* ]] || \
[[ $GPU == *"GeForce 830M"* ]] || \
[[ $GPU == *"GeForce 820M"* ]] || \
[[ $GPU == *"GeForce 800M"* ]] || \
[[ $GPU == *"GeForce GTX 780M"* ]] || \
[[ $GPU == *"GeForce GTX 770M"* ]] || \
[[ $GPU == *"GeForce GTX 765M"* ]] || \
[[ $GPU == *"GeForce GTX 760M"* ]] || \
[[ $GPU == *"GeForce GTX 680MX"* ]] || \
[[ $GPU == *"GeForce GTX 680M"* ]] || \
[[ $GPU == *"GeForce GTX 675MX"* ]] || \
[[ $GPU == *"GeForce GTX 675M"* ]] || \
[[ $GPU == *"GeForce GTX 670MX"* ]] || \
[[ $GPU == *"GeForce GTX 670M"* ]] || \
[[ $GPU == *"GeForce GTX 660M"* ]] || \
[[ $GPU == *"GeForce GT 755M"* ]] || \
[[ $GPU == *"GeForce GT 750M"* ]] || \
[[ $GPU == *"GeForce GT 650M	"* ]] || \
[[ $GPU == *"GeForce GT 745M"* ]] || \
[[ $GPU == *"GeForce GT 645M"* ]] || \
[[ $GPU == *"GeForce GT 740M"* ]] || \
[[ $GPU == *"GeForce GT 730M"* ]] || \
[[ $GPU == *"GeForce GT 640M"* ]] || \
[[ $GPU == *"GeForce GT 640M LE"* ]] || \
[[ $GPU == *"GeForce GT 735M"* ]] || \
[[ $GPU == *"GeForce GT 635M"* ]] || \
[[ $GPU == *"GeForce GT 730M"* ]] || \
[[ $GPU == *"GeForce GT 630M"* ]] || \
[[ $GPU == *"GeForce GT 625M"* ]] || \
[[ $GPU == *"GeForce GT 720M"* ]] || \
[[ $GPU == *"GeForce GT 620M	"* ]] || \
[[ $GPU == *"GeForce 710M"* ]] || \
[[ $GPU == *"GeForce 705M"* ]] || \
[[ $GPU == *"GeForce 610M"* ]] || \
[[ $GPU == *"GeForce GTX 580M"* ]] || \
[[ $GPU == *"GeForce GTX 570M"* ]] || \
[[ $GPU == *"GeForce GTX 560M"* ]] || \
[[ $GPU == *"GeForce GT 555M"* ]] || \
[[ $GPU == *"GeForce GT 550M"* ]] || \
[[ $GPU == *"GeForce GT 540M"* ]] || \
[[ $GPU == *"GeForce GT 525M"* ]] || \
[[ $GPU == *"GeForce GT 520MX"* ]] || \
[[ $GPU == *"GeForce GT 520M"* ]] || \
[[ $GPU == *"GeForce GTX 485M"* ]] || \
[[ $GPU == *"GeForce GTX 470M"* ]] || \
[[ $GPU == *"GeForce GTX 460M"* ]] || \
[[ $GPU == *"GeForce GT 445M	"* ]] || \
[[ $GPU == *"GeForce GT 435M"* ]] || \
[[ $GPU == *"GeForce GT 420M"* ]] || \
[[ $GPU == *"GeForce GT 415M"* ]] || \
[[ $GPU == *"GeForce GTX 480M"* ]] || \
[[ $GPU == *"GeForce 710M"* ]] || \
[[ $GPU == *"GeForce 410M"* ]] || \
[[ $GPU == *"GeForce GTX 1060 Ti"* ]] || \
[[ $GPU == *"GeForce GTX 1060"* ]]
then
    echo "CUDA supported GPU founded!"
    GPU=true
else
    echo "CUDA supported GPU could not be found!"
    GPU=false
fi

# Installing paddle and dependencies dependent from platform:
if [ "$AVX" = true ] && [ "$GPU" = true ]
then
    # Install paddle with gpu support.
    # Need codatoolkit (recomended version is 11.0 or higher, if conda can install it).
    # And cudnn, that installed from conda-forge channel, because just conda have only 7.6.5 max version.
    echo "Installing GPU and AVX version of paddle and dependencies."
    conda install -y paddlepaddle-gpu==2.1.0 cudatoolkit=11.2 -c paddle -c conda-forge   && \
    conda install -y -c conda-forge cudnn
    # Install dependencies for Autophoto:
    conda install -y pip  && \
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install nvidia-ml-py3
    # Add PATH and LD_CONFIG variables:
    echo 'export PATH=/opt/conda/include${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/opt/conda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    echo 'export CUDA_PATH=/opt/conda/' >> ~/.bashrc
    echo 'export CUDA_HOME=/opt/conda/' >> ~/.bashrc
    source ~/.bashrc
    # fix bad library files -_- cuda its cuda
    ln -s /opt/conda/lib/libcusolver.so.11 /opt/conda/lib/libcusolver.so.11.1
    ln -s /opt/conda/lib/libnvrtc-builtins.so.11.2 /opt/conda/lib/libnvrtc-builtins.so.11.1
    ln -s /opt/conda/lib/libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so.11.1
elif [ "$AVX" = false ] && [ "$GPU" = true ]
then
    echo "Installing GPU and noAVX version of paddle and dependencies."
    # Install paddle with GPU support and noAVX.
    # Need codatoolkit (recomended version is 11.0 or higher, if conda can install it).
    # And cudnn, that installed from conda-forge channel, because just conda have only 7.6.5 max version.
    conda install -y cudatoolkit=11.2 -c conda-forge
    conda install -y -c conda-forge cudnn
    conda install -y pip  && \
    pip install pillow  && \
    pip install numpy==1.20 && \
    pip install astor && \
    pip install requests && \
    pip install protobuf && \
    pip install six && \
    pip install decorator==4.4.2 && \
    pip install gast==0.4.0 && \
    pip install paddlepaddle-gpu==2.1.1 -f http://www.paddlepaddle.org.cn/whl/mkl/stable/noavx.html --no-index --trusted-host www.paddlepaddle.org.cn
    # Install dependencies for Autophoto:
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install nvidia-ml-py3
    # Add PATH and LD_CONFIG variables:
    echo 'export PATH=/opt/conda/include${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/opt/conda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    echo 'export CUDA_PATH=/opt/conda/' >> ~/.bashrc
    echo 'export CUDA_HOME=/opt/conda/' >> ~/.bashrc
    source ~/.bashrc
    # fix bad library files -_- cuda its cuda
    ln -s /opt/conda/lib/libcusolver.so.11 /opt/conda/lib/libcusolver.so.11.1
    ln -s /opt/conda/lib/libnvrtc-builtins.so.11.2 /opt/conda/lib/libnvrtc-builtins.so.11.1
    ln -s /opt/conda/lib/libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so.11.1
elif [ "$AVX" = true ] && [ "$GPU" = false ]
then
    echo "Installing only CPU and AVX version of paddle and dependencies."
    conda install -y paddlepaddle -c paddle
    # Install dependencies for Autophoto:
    conda install -y pip  && \
    pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
elif [ "$AVX" = false ] && [ "$GPU" = false ]
then
    echo "Installing only CPU and noAVX version of paddle and dependencies."
    # Install paddle only CPU without AVX:
    conda install -y pip  && \
    pip install pillow  && \
    pip install numpy==1.20 && \
    pip install astor && \
    pip install requests && \
    pip install protobuf && \
    pip install six && \
    pip install decorator==4.4.2 && \
    pip install gast==0.4.0 && \
    pip install paddlepaddle==2.1.1 -f http://www.paddlepaddle.org.cn/whl/mkl/stable/noavx.html --no-index --trusted-host www.paddlepaddle.org.cn
    # Install dependencies for Autophoto:
    pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
fi

# Install dependencies for Autophoto:
pip install paddlehub
cd /app
cd Detection
pip install -e .
cd ../..
