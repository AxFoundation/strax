import strax
from .plugin import Plugin

export, __all__ = strax.exporter()


@export
class ParallelSourcePlugin(Plugin):
    """An plugin that inlines the computations of other plugins and the saving of their results.

    This evades data transfer (pickling and/or memory copy) penalties while multiprocessing.

    """

    parallel = "process"
    # should we set this here?
    input_timeout = 300

    @classmethod
    def inline_plugins(cls, components, start_from, log):
        plugins = components.plugins.copy()
        loader_plugins = components.loader_plugins.copy()
        log.debug(f"Try to inline plugins starting from {start_from}")

        sub_plugins = {start_from: plugins[start_from]}
        del plugins[start_from]

        # Gather all plugins that do not rechunk and which branch out as a
        # simple tree from the input plugin.
        # We'll run these all together in one process.
        while True:
            # Scan for plugins we can inline
            for p in plugins.values():
                if p.parallel and all([d in sub_plugins for d in p.depends_on]):
                    for d in p.provides:
                        sub_plugins[d] = p
                        if d in plugins:
                            del plugins[d]
                    # Rescan
                    break
            else:
                # No more plugins we can inline
                break
        log.debug(f"Trying to inline the following sub-plugins: {sub_plugins}")
        if len(set(list(sub_plugins.values()))) == 1:
            # Just one plugin to inline: no use
            log.debug("Just one plugin to inline: skipping")
            return components

        # Which data types should we output? Three cases follow.
        outputs_to_send = set()

        # Case 1. Requested as a final target
        for p in sub_plugins.values():
            outputs_to_send.update(set(components.targets).intersection(set(p.provides)))
        # Case 2. Requested by a plugin we did not inline
        for d, p in plugins.items():
            outputs_to_send.update(set(p.depends_on))
        outputs_to_send &= sub_plugins.keys()

        # Inline savers that do not require rechunking
        savers = components.savers
        sub_savers = dict()
        for p in sub_plugins.values():
            for d in p.provides:
                if d not in savers:
                    continue
                if p.can_rechunk(d):
                    # Case 3. has a saver we can't inline
                    outputs_to_send.add(d)
                    continue

                remaining_savers = []
                for s_i, s in enumerate(savers[d]):
                    if not s.allow_fork:
                        # Case 3 again, cannot inline saver
                        outputs_to_send.add(d)
                        remaining_savers.append(s)
                        continue
                    if d not in sub_savers:
                        sub_savers[d] = []
                    s.is_forked = True
                    sub_savers[d].append(s)
                savers[d] = remaining_savers

                if not len(savers[d]):
                    del savers[d]

        p = cls(depends_on=sub_plugins[start_from].depends_on)
        p.run_id = sub_plugins[start_from]._run_id
        p.sub_plugins = sub_plugins
        assert len(outputs_to_send)
        p.provides = tuple(outputs_to_send)
        p.sub_savers = sub_savers
        p.start_from = start_from
        if p.multi_output:
            p.dtype = {}
            for d in outputs_to_send:
                if d in p.sub_plugins:
                    p.dtype[d] = p.sub_plugins[d].dtype_for(d)
                else:
                    log.debug(f"Finding plugin that provides {d}")
                    # Need to do some more work to get the plugin that
                    # provides this data-type.
                    for sp in p.sub_plugins.values():
                        if d in sp.provides:
                            log.debug(f"{sp} provides {d}")
                            p.dtype[d] = sp.dtype_for(d)
                            break
        else:
            to_send = list(outputs_to_send)[0]
            p.dtype = p.sub_plugins[to_send].dtype_for(to_send)
        for d in p.provides:
            plugins[d] = p

        log.debug(f"Trying to find plugins for dependencies: {p.depends_on}")

        p.deps = {
            d: plugins[d] if plugins.get(d, None) else loader_plugins[d] for d in p.depends_on
        }

        log.debug(f"Inlined plugins: {p.sub_plugins}.Inlined savers: {p.sub_savers}")

        return strax.ProcessorComponents(
            plugins, components.loaders, components.loader_plugins, savers, components.targets
        )

    def __init__(self, depends_on):
        self.depends_on = depends_on
        super().__init__()

    def source_finished(self):
        return self.sub_plugins[self.start_from].source_finished()

    def is_ready(self, chunk_i):
        return self.sub_plugins[self.start_from].is_ready(chunk_i)

    def do_compute(self, chunk_i=None, **kwargs):
        results = kwargs

        # Run the different plugin computations
        while True:
            for output_name, p in self.sub_plugins.items():
                if output_name in results:
                    continue
                if any([d not in results for d in p.depends_on]):
                    continue
                compute_kwargs = dict(chunk_i=chunk_i)

                for kind, d_of_kind in p.dependencies_by_kind().items():
                    compute_kwargs[kind] = strax.Chunk.merge([results[d] for d in d_of_kind])

                # Store compute result(s)
                r = p.do_compute(**compute_kwargs)
                if p.multi_output:
                    for d in r:
                        results[d] = r[d]
                else:
                    results[output_name] = r

                # Rescan plugins to see if we can compute anything more
                break

            else:
                # Nothing further to compute
                break
        for d in self.provides:
            assert d in results, f"Output {d} missing!"

        # Save anything we can through the inlined savers
        for d, savers in self.sub_savers.items():
            for s in savers:
                s.save(chunk=results[d], chunk_i=chunk_i)

        # Remove results we do not need to send
        for d in list(results.keys()):
            if d not in self.provides:
                del results[d]

        if self.multi_output:
            for k in self.provides:
                assert k in results
                assert isinstance(results[k], strax.Chunk)
                r0 = results[k]
        else:
            results = r0 = results[self.provides[0]]
            assert isinstance(r0, strax.Chunk)

        return self._fix_output(
            results, start=r0.start, end=r0.end, superrun=r0.superrun, subruns=r0.subruns
        )

    def cleanup(self, wait_for):
        print(f"{self.__class__.__name__} terminated. Waiting for {len(wait_for)} pending futures.")
        for savers in self.sub_savers.values():
            for s in savers:
                s.close(wait_for=wait_for)
        super().cleanup(wait_for)
