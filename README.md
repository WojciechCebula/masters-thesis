# masters

## Installation steps

1. Create new virtual environment:

    ```
    conda create --name masters python=3.10
    ```

2. Activate environment
    ```
    conda activate masters
    ```

3. Update _pip_ version:
    ```
    python -m pip install --upgrade pip
    ```
4. Install _ptp_ package:

    ```
    python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu121
    ```
5. Enable precommit hook:
    ```
    pre-commit install
    ```


## Useful sources

- https://github.com/ashleve/lightning-hydra-template
