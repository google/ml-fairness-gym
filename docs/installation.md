# Installation instructions

To install ML-fairness-gym:


```shell
git clone https://github.com/google/ml-fairness-gym
cd ml-fairness-gym
virtualenv -p python3 .
source ./bin/activate
pip install -r requirements.txt
```

Note that ML-fairness-gym should only be run with python 3.

Run the following command to add ml-fairness-gym to your PYTHONPATH.

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/ml-fairness-gym"
```

To check that everything installed correctly, you can run:

```shell
./tests.sh
```

## Troubleshooting
If pip is not installed. Follow [instructions to install it](https://pip.pypa.io/en/stable/installing/).

If virtualenv is not installed, you can install it with apt-get.

```shell
sudo apt-get install virtualenv
```

If you get an error: "ModuleNotFoundError: No module named 'tkinter'", you may
need to install tkinter.

```shell
sudo apt-get install python3-tk
```
