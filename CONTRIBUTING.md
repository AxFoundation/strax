## Contribution guidelines

You're welcome to contribute to strax! 

Currently many features are still in significant flux, and the documentation is still very basic. Until more people start getting involved in development, we're probably not even following our own advice below...

### Please fork
Please work in a fork, then submit pull requests. 
Only maintainers sometimes work in branches if there is a good reason for it.

### No large files
Avoid committing large (> 100 kB) files. We'd like to keep the repository no more than a few MB.

For example, do not commit jupyter notebooks with high-resolution plots (clear the output first), or long configuration files, or binary test data. 

While it's possible to rewrite history to remove large files, this is a bit of work and messes with the repository's consistency. Once data has gone to master it's especially difficult, then there's a risk of others merging the files back in later unless they cooperate in the history-rewriting.

This is one reason to prefer forks over branches; if you commit a huge file by mistake it's just in your fork.  

### Code style
Of course, please write nice and clean code :-)

PEP8-compatibility is great (you can test with flake8) but not as important as other good coding habits such as avoiding duplication. See e.g. the [famous beyond PEP8 talk](https://www.youtube.com/watch?v=wf-BqAjZb8M). 

In particular, don't go into code someone else is maintaining to "PEP8-ify" it (or worse, use some automatic styling tool)

Other style guidelines (docstrings etc.) are yet to be determined.

### Pull requests
When accepting pull requests, prefer merge or squash depending on how the commit history looks.
- If it's dozens of 'oops' and 'test' commits, best to squash.
- If it's a few commits that mostly outline discrete steps of an implementation, it's worth keeping, so best to merge.
