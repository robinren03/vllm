#!/bin/bash  

if [[ "$(ps -p $$ -ocomm=)" ==  "bash" && "$(basename -- "$0")" == "-bash" ]]; then  
	    echo "Configured"  
    else  
	        echo "Error! Please use source to run the script!"  
		    return 1 2>/dev/null || exit 1
fi  

export CUDA_HOME=/usr/local/cuda-12.1  
export PATH="${CUDA_HOME}/bin:$PATH"  
export http_proxy="127.0.0.1:8889"  
export https_proxy="127.0.0.1:8889"
