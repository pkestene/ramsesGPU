#!/bin/bash 
./euler2d_cpu_qt --posx 200 --posy 100&
./euler2d_gpu_qt --posx 460 --posy 100&


# NOTE1 : to make a screencast of test, before running this screen, first launch
# recordmydesktop -x 200 -y 100 --width 430 --height 840 -o euler_cpu_gpu_screencast.ogv

# NOTE2 : to re-encode ogv video into avi:
# mencoder in.ogv -ovc lavc -nosound -lavcopts vcodec=mpeg4:vqscale=4 -ffourcc DX50 -o out.avi
