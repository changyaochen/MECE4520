## Python

In data science and machine learning, the de-facto official language is [Python](https://en.wikipedia.org/wiki/Python_(programming_language)).
For this course, we will use Python as the primary language (*e.g.,* for examples and homework solutions).
If you choose other programming languages, *e.g.*, Matlab, R, it won't be a blocker, but we will not provide support for it.

### Jupyter
Throughout the course, we will use [Jupyter](https://jupyter.org/) to run Python codes. With Jupyter, you can run Python codes and view the results from a web browser.
There are multiple methods to configure the programming environment in order to run Jupyter notebook(s), as described below.

## Programming environment configuration

### Google Colab (Recommended)
To set up a Python/Jupyter environment for this course, we recommend the students using [Google Colab](https://colab.research.google.com/).
With Google Colab, you don't need to configure anything on your local machine.
[This](https://www.youtube.com/watch?v=inN8seMm7UI) is an introduction to Google Colab.

All of the Jupyter notebook examples used in the class will be pushed to Github, and you can run those notebooks directly from Google Colab.


### Local development with Docker
[Docker](https://www.docker.com/) is a framework to allow for portable and reproducible programing environments. For this course, we provide a Docker image to launch a Jupyter server on your local machine.

1. Install [Git](https://git-scm.com/downloads) and [Docker Engine](https://docs.docker.com/engine/install/).
2. Pull this repository to your local machine, by running command in terminal:
    ```sh
    cd <directory_of_your_choice>
    git clone git@github.com:changyaochen/MECE4520.git
    ```
3. Run the following commands in terminal:
   ```sh
   cd MECE4520
   make build  # only needed for the first-time use, it can take a few minutes
   make serve
   ```
4. If things go smoothly, you should see many lines printed to the terminal. Open a browser, and go to `localhost:8888`. It should show you the JupyterLab interface.
5. To shut down the Jupyter server, go back to the terminal session where you launch the server, and press "cmd + C" ("control + C" for Windows and Linux).

## Troubleshooting

Try Google with the error message, usually someone already has the answer.
