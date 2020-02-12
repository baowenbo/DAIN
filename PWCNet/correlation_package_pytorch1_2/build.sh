
#!/usr/bin/env bash

echo "Need pytorch>=1.0.0"
conda activate pytorch1.2.0


rm -rf build *.egg-info dist
python setup.py install

