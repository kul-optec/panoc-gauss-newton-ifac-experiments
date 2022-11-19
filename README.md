# Gauss–Newton meets PANOC: A fast and globally convergent algorithm for nonlinear optimal control

To reproduce (Linux, requires Python 3, CMake, Ninja, a modern C/C++ toolchain):

```sh
# Create a Python virtual environment
python3 -m venv py-venv
. py-venv/bin/activate
# Set compiler flags for optimal performance
export CFLAGS=-march=native
export CXXFLAGS=-march=native
# Install alpaqa dependencies into virtual environment
wget https://raw.githubusercontent.com/kul-optec/alpaqa/50ea3edaa6f3c79cb10f3f7816ef475606cd11c8/scripts/install-casadi-static.sh -O- | bash
wget https://raw.githubusercontent.com/kul-optec/alpaqa/50ea3edaa6f3c79cb10f3f7816ef475606cd11c8/scripts/install-eigen.sh -O- | bash
# Install Python dependencies, build alpaqa from source
pip install -r requirements.txt # takes a couple of minutes
# Run the experiments and generate the figures
make # takes some more minutes, close the figures to start next experiment
```
