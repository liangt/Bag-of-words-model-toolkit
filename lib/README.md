# Dense SIFT descriptor

Implement the dense SIFT descriptor based on VLfeat library and compile it to dynamic library; Then use ctypes interface to call it in python.

How to use:
1. Download VLfeat library;
2. Compile dsift.c;
   for example, in 64-bit Ubuntu 14.04,
   g++ dsift.c -fPIC -shared -o libdsift.so -LVLfeat_root/bin/glnxa64/libvl.so -IVLfeat_root/vl 
