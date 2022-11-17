sudo mkfs.ext4 /dev/pmem1
sudo mount -t ext4 -o dax /dev/pmem1 /mnt/pmem1