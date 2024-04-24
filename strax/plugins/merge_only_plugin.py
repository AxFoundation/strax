import strax
from .plugin import Plugin, SaveWhen

export, __all__ = strax.exporter()


##
# "Plugins" for internal use
# These do not actually do computations, but do other tasks
# for which posing as a plugin is helpful.
# Do not subclass unless you know what you are doing..
##


@export
class MergeOnlyPlugin(Plugin):
    """Plugin that merges data from its dependencies."""

    save_when = SaveWhen.EXPLICIT

    def infer_dtype(self):
        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError(
                "MergeOnlyPlugins can only merge data of the same kind, but got multiple kinds: "
                + str(deps_by_kind)
            )

        return strax.merged_dtype(
            [
                self.deps[d].dtype_for(d)
                # Sorting is needed here to match what strax.Chunk does in merging
                for d in sorted(self.depends_on)
            ]
        )

    def compute(self, **kwargs):
        return kwargs[list(kwargs.keys())[0]]
