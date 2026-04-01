# What is "uv" in python?

`uv` is a modern, super-fast Python `package manager` and `environment tool` — basically an all-in-one replacement for:

- pip (installing packages)
- venv (creating virtual environments)
- pip-tools (dependency management)

# how to use "UV"

```python
$ pip install uv
$ uv init
$ uv venv
$ vim requirement.txt
$ uv pip install -r requirement.txt
    or
$ uv add -r requirements.txt
$ uv run main.py
```

# why uv is popular

- Extremely fast (written in Rust)
- Manages dependencies + environments together
- Automatically creates lockfiles (like package-lock.json)
- No need to manually activate virtual environments
- Cleaner workflow


`-- made by Ro706`
