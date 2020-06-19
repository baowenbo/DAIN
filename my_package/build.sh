#!/usr/bin/env bash

echo "Need pytorch>=1.0.0"
source activate pytorch1.0.0

export PYTHONPATH=$PYTHONPATH:$(pwd)

cd MinDepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConv
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd InterpolationCh
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd DepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd Interpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConvFlow
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FilterInterpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

