# References: https://developer.nvidia.com/cuda-gpus
nvcc_args = [
    # Tesla: K80, K80
    # Quadro: (None)
    # NVIDIA NVS: (None)
    # Jetson: (None)
    '-gencode', 'arch=compute_37,code=sm_37',

    # Tesla: (None)
    # Quadro: K1200, K620, M1200, M520, M5000M, M4000M, M3000M, M2000M, M1000M, K620M, M600M, M500M
    # NVIDIA NVS: 810
    # GeForce / Titan: GTX 750 Ti, GTX 750, GTX 960M, GTX 950M, 940M, 930M, GTX 860M, GTX 850M, 840M, 830M
    # Jetson: (None)
    '-gencode', 'arch=compute_50,code=sm_50',

    # Tesla: M60, M40
    # Quadro: M6000 24GB, M6000, M5000, M4000, M2000, M5500M, M2200, M620
    # NVIDIA NVS: (None)
    # GeForce / Titan: GTX TITAN X, GTX 980 Ti, GTX 980, GTX 970, GTX 960, GTX 950, GTX 980, GTX 980M, GTX 970M, GTX 965M, 910M
    # Jetson: (None)
    '-gencode', 'arch=compute_52,code=sm_52',

    # Tesla: P100
    # Quadro: GP100
    # NVIDIA: NVS: (None)
    # GeForce / Titan: (None)
    # Jetson: (None)
    '-gencode', 'arch=compute_60,code=sm_60',

    # Tesla: P40, P4
    # Quadro: P6000, P5000, P4000, P2200, P2000, P1000, P620, P600, P400, P620, P520, P5200, P4200, P3200, P5000, P4000, P3000, P2000, P1000, P600, P500
    # NVIDIA NVS: (None)
    # GeForce / Titan: TITAN Xp, TITAN X, GTX 1080 Ti, GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1080, GTX 1070, GTX 1060
    # Jetson: (None)
    '-gencode', 'arch=compute_61,code=sm_61',

    # Tesla: T4
    # Quadro: RTX 8000, RTX 6000, RTX 5000, RTX 4000, RTX 5000, RTX 4000, RTX 3000, T2000, T1000
    # NVIDIA NVS: (None)
    # GeForce / Titan: TITAN RTX, RTX 2080 Ti, RTX 2080, RTX 2070, RTX 2060, RTX 2080, RTX 2070, RTX 2060
    # Jetson: (None)
    '-gencode', 'arch=compute_75,code=sm_75',

    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_70,code=compute_70'

    '-w' # Ignore compiler warnings.
]

cxx_args = ['-std=c++11', '-w']