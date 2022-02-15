## Transfer Learning with Transformers :handshake: BentoML

### Installation

Run [fine_tune_roberta.sync.ipynb](./fine_tune_roberta.sync.ipynb) either locally or on [Colab](https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb)

You can either train the model on Colab and then import it into BentoML, or
quickly import our pretrained model with `bentoml import` CLI:
```bash
# TODO: when import/export CLI is ready
```


### Development
Uses [jupyter_ascending](https://github.com/untitled-ai/jupyter_ascending.vim), managed with [wbthomason/packer.nvim](https://github.com/wbthomason/packer.nvim) for [Neovim](https://neovim.io/) for development.

Refers to [Installation](https://github.com/untitled-ai/jupyter_ascending.vim#installation) for how to setup the plugins correctly.

```python
# Run the below in THIS DIRECTORY:
jupyter notebook fine_tune_roberta.sync.ipynb
```

<details>

    <summary>Neovim</summary>

    If you are using VimScript, then follow instruction [here](https://github.com/untitled-ai/jupyter_ascending.vim#installation).

    If you are using Lua within Neovim, add the following under `init.lua`:

    ```lua
    local vim = vim

    vim.g.jupyter_ascending_auto_write = 'true'

    vim.api.nvim_set_keymap('n', '<space><space>x', '<CR>:call jupyter_ascending#execute()<CR>', {})
    vim.api.nvim_set_keymap('n', '<space><space>X', '<CR>:call jupyter_ascending#execute_all()<CR>', {})
    ```

    Save and recompile to source the changes in configuration.

    Then edit [fine_tune_roberta.sync.py](./fine_tune_roberta.sync.py). The jupyter notebook will be synced whenever you saved the python file.

    <b>NOTE: </b> The Lua configuration is opinionated, meaning you can customize to your usage. The above is just an example on how you can achieve the equivalent setup in VimScript

</details>

<details>

    <summary>Terminal</summary>

    If you just want to use python API of `jupyter_ascending`:

    1. Edit [`fine_tune_roberta.sync.py`](./fine_tune_roberta.sync.py)
    2. Sync the code into jupyter notebook: `python -m jupyter_ascending.requests.sync --filename fine_tune_roberta.sync.py`
    3. Run the cell code: `python -m jupyter_ascending.requests.execute --filename fine_tune_roberta.sync.py --line 16`

</details>
