# Python environment setup

Python environment management is known to be notorious confusing (_e.g._, see [here](https://www.explainxkcd.com/wiki/index.php/1987:_Python_Environment)). At the heart of it, we have to specify:
1. Python interpreter version.
2. Python library versions (_e.g._, `numpy`).


## Virtual environment management
To handle both issues at once, we use a tool called `pipenv` ([link](https://pipenv.pypa.io/en/latest/)). Please follow the [instruction](https://pipenv.pypa.io/en/latest/#install-pipenv-today) on the website to complete the installation.

In our course, we highly recommend to use Python interpreter with version >= 3.7. (_e.g._, 3.7.13). For the Python libraries, please refer to the `Pipeflie` file.

To use `pipenv`, we will create a new virtual environment and run our codes within. To do so: open a new session in Terminal, and run:
```sh
# create the virtual environment using the Pipfile.lock
pipenv install
```
To activate the virtual environment, simply run:
```sh
pipenv shell
```
To deactivate (exit) the virtual environment, run:
```sh
exit
```

## Jupyter Notebook
Through out the course, we will use [Jupyter Notebook](https://jupyter.org/) to run various Python codes. To launch the Jupyter notebook server, first start the `pipenv` virtual environment as outlined in the previous section, and then run:
```sh
jupyter notebook
```
The Jupyter notebook should open with your default browser (usually at `http://localhost:8888/`). To shut it down, go back to the terminal session where you launch the Jupyter notebook server, and press "cmd + C" ("control + C" for Windows and Linux).

## Troubleshooting

Try Google with the error message, usually someone already has the answer.

### `command not found: pip`
It is likely the default Python interpreter on your machine is not Python 3.x. Try using `pip3` in place of `pip`

### Warning message when installing `pipenv`
During the installation of `pipenv`, if you see the following warning message:

> WARNING: The scripts pipenv and pipenv-resolver are installed in '/path/to/bin' which is not on PATH

Then open your `~/.bash_profile` or `~/.zshrc` file with a text editor, and add the following line to the end:
```sh
export PATH=/path/to/bin:$PATH
```
Note that here `/path/to/bin` is just a placeholder and you will replace it with the actual string you see. Then start a new shell session in the terminal, and you should be able to use `pipenv` normally. To check, run the following in the terminal:
```sh
which pipenv
```
and you should see the path to the `pipenv` binary.
