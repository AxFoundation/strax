# Rather boring __main__, makes it possible to test if strax imports with
# python -m strax
import strax    # noqa
print(f"Strax version {strax.__version__} says hi!")
