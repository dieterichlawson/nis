#!/bin/bash

LOGDIR=$1
TAG=$2
mkdir .tmp_ims
echo "Extracting images from TB summaries"
python3 save_images.py --logdir=$LOGDIR --tag=$TAG --outdir=.tmp_ims
echo "Labelling images with step"
ls .tmp_ims | parallel 'convert .tmp_ims/{} -scale 200% -quality 100% -compress LossLess -background Black -fill white -pointsize 10 label:{= $_ =~
/image_0*(\d*).png/; $_ = $1; =} -gravity Center -append .tmp_ims/{.}_lb.png'
echo "Making gif"
convert -delay 2 -loop 0 -colors 256 -compress LossLess -quality 100% .tmp_ims/image_0*_lb.png out.gif
rm -rf .tmp_ims
