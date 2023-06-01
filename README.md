# Pash

## Dependencies
* gcc: 7.5.0
* cmake: 3.10.2
* C++ boost library

## Compile
```
    mkdir build
    cd build
    cmake ..
    make
```

## PM environment
```
    sh mount.sh
    sudo dd if=/dev/zero of=/mnt/pmem1/btree bs=1048576 count=num-MB
    sh create_pm_files.sh
```

## RUN
### Options
```
    -t: Thread number (Default: 1)
    -n: Load number (Default: 0)
    -p: Operation number (Default: 20000000)
    -op: Operation type (Default: full, option: insert/pos/neg/delete/mixed)
    -distribution: Workload distribution (Default: uniform, option: uniform/skew)
    -skew: Workload skewness factor (Default: 0.99)
    -bs: Pipeline batch size (Default: 4)
```
### Single thread evaluation
```
    ./test_pmem -n ${load_num} -p ${op_num}
```

### Multi-threaded evaluation
```
    ./test_pmem -n ${load_num} -p ${op_num} -t ${thread_num}
```

### YCSB
```
    ./test_pmem -n ${load_num} -p ${parallel_load_num} -u ${mixed_op_num} -t ${thread_num} -distribution skew -skew ${skew_factor} -op mixed -r ${read_ratio} -s ${insert_ratio}
```