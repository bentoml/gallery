## Fine-tune with Transformers :handshake: BentoML

<div align='center'>
    <p align='center'>
        <a href="https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        <a href="https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="nbviewer"/></a>
        <a href="https://github.com/bentoml/gallery/tree/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter" alt="Made with Jupyter"/></a>
    </p>
</div>

### How to use this

One can fine tune the model by running [notebook](./fine_tune_roberta.sync.ipynb) or import our [fine-tune model](./exported) under a IPython shell:
```python
import bentoml
tag = bentoml.models.import_model("./exported")
model, tokenizer = bentoml.transformers.load(tag, return_config=False)
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


<b>Experimental:</b> run `./run_fine_tune` under this directory.
