# Building Jupyter book

## When you first clone the repository
You should create a virtual environment (I recommend using python instead of conda) and activate it:

```bash
python3 -m venv book
source book/bin/activate
```

Install `jupyter-books` and `ghp-import`

```bash
pip install jupyter-book ghp-import
```

You can now build the book with the command
```bash
jupyter-book build --all
```

## Any other time
Activate the virtual enviroment, make your edits to `book.md` and build the book artifacts.

## Viewing your book
After the build step, the output of the terminal will include a link to the `index.html` file that was built.
You can paste this file path into your browser to render the book.

## Publishing book
Make sure you have built all the latest changes to the book.
There is a convenience script, `ghp-upload.sh` that you should run using
```bash
sh ghp-upload.sh
```
