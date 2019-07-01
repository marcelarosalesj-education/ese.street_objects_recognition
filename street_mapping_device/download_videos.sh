#!/bin/bash

input_dir="input" 
mkdir -p "$input_dir"

link="https://uc2136c91a73cea6218ba9654b84.dl.dropboxusercontent.com/cd/0/get/Aj3c_sSHM3eOMYGv81j4bFWTe_M4ppYYrjtBgolxSYRqT8Ru2iG-UOblD_HDQFIjHiQ20RcxGS-ANeX8_w21DZvKoF6uudKI-KQo6PM-W0sI_A/file?dl=1#"
tarball="videos.tar.gz"
wget -O "$input_dir/$tarball" $link


pushd "$input_dir"
tar -xvzf $tarball
cd ..
