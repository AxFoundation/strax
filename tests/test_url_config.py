"""Tests for the URLConfig engine in strax.url_config.

strax itself does not ship any protocols, so the tests register their own on a subclass with
separate registries (``_LOOKUP`` / ``_PREPROCESSORS``). This keeps the test protocols out of the
global ``strax.URLConfig`` registry that downstream packages (e.g. straxen) register into.

"""

import json
import pickle
import random
import unittest

import numpy as np
import pandas as pd

import strax


class DummyObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class URLConfigForTests(strax.URLConfig):
    """URLConfig with its own protocol and preprocessor registries."""

    _LOOKUP: dict = {}
    _PREPROCESSORS = ()


@URLConfigForTests.register("json")
def read_json(content: str, **kwargs):
    return json.loads(content)


@URLConfigForTests.register("format")
def format_arg(arg: str, **kwargs):
    return arg.format(**kwargs)


@URLConfigForTests.register("take")
def get_key(container, take=None, **kwargs):
    if take is None:
        return container
    if not isinstance(take, list):
        take = [take]
    for t in take:
        container = container[t]
    return container


@URLConfigForTests.register("random")
def generate_random(_):
    return random.random()


@URLConfigForTests.register("unpicklable")
def return_lambda(_):
    return lambda x: x


@URLConfigForTests.register("large-array")
def large_array(_):
    return np.ones(1_000_000).tolist()


@URLConfigForTests.register("only-a")
def only_a(arg, a=None):
    return (arg, a)


@URLConfigForTests.preprocessor
def formatter(config, **kwargs):
    if not isinstance(config, str):
        return config
    try:
        config = config.format(**kwargs)
    except KeyError:
        pass
    return config


GLOBAL_VERSIONS = {"global_v1": {"test_config": "v0"}}


@URLConfigForTests.preprocessor
def replace_global_version(config, name=None, **kwargs):
    if name is None:
        return

    if not isinstance(config, str):
        return config

    if URLConfigForTests.SCHEME_SEP not in config:
        return config

    version = URLConfigForTests.kwarg_from_url(config, "version")

    if version is None:
        return config

    if version.startswith("global_") and version in GLOBAL_VERSIONS:
        version = GLOBAL_VERSIONS[version].get(name, version)
        config = URLConfigForTests.format_url_kwargs(config, version=version)
    return config


class ExamplePlugin(strax.Plugin):
    depends_on = ()
    dtype = strax.time_fields
    provides = "test_data"
    test_config = URLConfigForTests(default=42)
    cached_config = URLConfigForTests(default=666, cache=1)
    typed_config = URLConfigForTests(default=1, type=int)


RUN_ID = "000001"


class TestURLConfig(unittest.TestCase):
    def setUp(self):
        self.st = strax.Context(storage=[], register=[ExamplePlugin])

    def get_plugin(self):
        return self.st.get_single_plugin(RUN_ID, "test_data")

    # --- Exports and registries ---

    def test_exported(self):
        assert issubclass(strax.URLConfig, strax.Config)
        assert callable(strax.clear_config_caches)
        assert callable(strax.config_cache_size_mb)

    def test_registry_isolated(self):
        """Test protocols must not leak into the global strax.URLConfig registry."""
        assert "random" in URLConfigForTests._LOOKUP
        assert "random" not in strax.URLConfig._LOOKUP
        assert formatter not in [f for _, f in strax.URLConfig._PREPROCESSORS]

    def test_register_duplicate(self):
        with self.assertRaises(ValueError):
            URLConfigForTests.register("json", read_json)

    def test_register_non_string(self):
        with self.assertRaises(ValueError):
            URLConfigForTests.register(42, read_json)

    def test_register_tuple(self):
        class LocalURLConfig(strax.URLConfig):
            _LOOKUP: dict = {}
            _PREPROCESSORS = ()

        @LocalURLConfig.register(("proto-a", "proto-b"))
        def ident(arg):
            return arg

        assert LocalURLConfig._LOOKUP["proto-a"] is ident
        assert LocalURLConfig._LOOKUP["proto-b"] is ident

    def test_preprocessor_duplicate(self):
        with self.assertRaises(ValueError):
            URLConfigForTests.preprocessor(formatter)

    def test_preprocessor_precedence(self):
        """Preprocessors with higher precedence run first."""
        calls = []

        class LocalURLConfig(strax.URLConfig):
            _LOOKUP: dict = {}
            _PREPROCESSORS = ()

        @LocalURLConfig.preprocessor(precedence=0)
        def low(config):
            calls.append("low")
            return config + "-low"

        @LocalURLConfig.preprocessor(precedence=10)
        def high(config):
            calls.append("high")
            return config + "-high"

        class LocalPlugin(ExamplePlugin):
            local_config = LocalURLConfig(default="x")

        st = strax.Context(storage=[], register=[LocalPlugin])
        p = st.get_single_plugin(RUN_ID, "test_data")
        # validate may run more than once per plugin; each pass runs high before low
        self.assertEqual(calls[:2], ["high", "low"])
        self.assertEqual(calls, ["high", "low"] * (len(calls) // 2))
        self.assertEqual(p.local_config, "x-high-low")

    # --- Fetching values through a plugin ---

    def test_default(self):
        p = self.get_plugin()
        self.assertEqual(p.test_config, 42)

    def test_literal(self):
        self.st.set_config({"test_config": 666})
        p = self.get_plugin()
        self.assertEqual(p.test_config, 666)

    def test_unknown_protocol_is_literal(self):
        self.st.set_config({"test_config": "unknown://abc"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, "unknown://abc")

    def test_leading_zero_int(self):
        self.st.set_config({"test_config": "format://{value}?value=0666"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, "0666")
        self.assertIsInstance(p.test_config, str)

    def test_json_protocol(self):
        self.st.set_config({"test_config": 'json://{"a":0}'})
        p = self.get_plugin()
        self.assertEqual(p.test_config, {"a": 0})

    def test_plugin_attribute_lookup(self):
        self.st.set_config({"test_config": "format://{run_id}?run_id=plugin.run_id"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, RUN_ID)

    def test_config_lookup(self):
        self.st.set_config({"test_config": "format://{v}?v=config.cached_config"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, "666")

    def test_take_nested(self):
        self.st.set_config(
            {
                "test_config": (
                    'take://json://{"a":{"aa":0,"ab":1},"b":{"ba":2,"bb":3}}?take=b&take=ba'
                )
            }
        )
        p = self.get_plugin()
        self.assertEqual(p.test_config, 2)

    def test_eval_filters_kwargs(self):
        """Kwargs that the protocol does not accept are dropped."""
        self.assertEqual(URLConfigForTests.eval("only-a://x?a=1&b=2"), ("x", 1))

    def test_format_preprocessor(self):
        self.st.set_config({"test_config": "{name}:{run_id}"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, f"test_config:{RUN_ID}")
        self.assertEqual(p.test_config, p.config["test_config"])

    def test_global_version_preprocessor(self):
        self.st.set_config({"test_config": "fake://url?version=global_v1"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, "fake://url?version=v0")

    def test_global_version_not_changed(self):
        """
        - if no global version is matched, the url version should not be changed
        - if config is not matched, the url version should not be changed
        """
        assert "global_v2" not in GLOBAL_VERSIONS
        self.st.set_config({"test_config": "fake://url?version=global_v2"})
        p = self.get_plugin()
        self.assertEqual(p.test_config, "fake://url?version=global_v2")

        class ExamplePluginNew(ExamplePlugin):
            test_config_new = URLConfigForTests(default="rootisthesourceofallevil")

        self.st.register(ExamplePluginNew)
        self.st.set_config({"test_config_new": "fake://url?version=global_v1"})
        p = self.get_plugin()
        self.assertEqual(p.test_config_new, "fake://url?version=global_v1")

    def test_url_is_hashed_not_value(self):
        """The lineage contains the (preprocessed) URL, not the fetched value."""
        url = 'take://json://{"a":1}?take=a'
        self.st.set_config({"test_config": url})
        lineage = self.st.key_for(RUN_ID, "test_data").lineage
        self.assertEqual(lineage["test_data"][2]["test_config"], url)

    # --- Type validation ---

    def test_type_not_enforced_on_url(self):
        self.st.set_config({"typed_config": 'json://"2"'})
        p = self.get_plugin()
        # pydantic coerces "2" -> 2 is allowed, the value is returned as-is
        self.assertEqual(p.typed_config, "2")

    def test_type_validated_on_fetched_value(self):
        self.st.set_config({"typed_config": 'json://"abc"'})
        p = self.get_plugin()
        with self.assertRaises(TypeError):
            p.typed_config

    def test_type_validated_on_literal(self):
        self.st.set_config({"typed_config": "abc"})
        with self.assertRaises(TypeError):
            self.get_plugin()

    def test_type_tuple(self):
        class LocalPlugin(ExamplePlugin):
            tuple_typed = URLConfigForTests(default=1, type=(int, dict))

        st = strax.Context(storage=[], register=[LocalPlugin])
        st.set_config({"tuple_typed": 'json://{"a":1}'})
        self.assertEqual(st.get_single_plugin(RUN_ID, "test_data").tuple_typed, {"a": 1})
        st.set_config({"tuple_typed": 'json://"abc"'})
        with self.assertRaises(TypeError):
            st.get_single_plugin(RUN_ID, "test_data").tuple_typed

    # --- Caching ---

    def test_cache(self):
        p = self.get_plugin()

        # sanity check that default value is not affected
        self.assertEqual(p.cached_config, 666)
        self.st.set_config({"cached_config": "random://abc"})
        p = self.get_plugin()

        # value is randomly generated when accessed so if
        # its equal when we access it again, its coming from the cache
        cached_value = p.cached_config
        self.assertEqual(cached_value, p.cached_config)

        # now change the config to which will generate a new number
        self.st.set_config({"cached_config": "random://dfg"})
        p = self.get_plugin()

        # sanity check that the new value is still consistent i.e. cached
        self.assertEqual(p.cached_config, p.cached_config)

        # test if previous value is evicted, since cache size is 1
        self.assertNotEqual(cached_value, p.cached_config)

        # verify pickalibility of objects in cache dont affect plugin pickalibility
        self.st.set_config({"cached_config": "unpicklable://dfg"})
        p = self.get_plugin()
        with self.assertRaises(AttributeError):
            pickle.dumps(p.cached_config)
        pickle.dumps(p)

    def test_uncached(self):
        self.st.set_config({"test_config": "random://abc"})
        p = self.get_plugin()
        self.assertNotEqual(p.test_config, p.test_config)

    def test_cache_size(self):
        """Test the cache helper functions."""
        # make sure the value has a detectable size
        self.st.set_config({"cached_config": "large-array://dfg"})
        p = self.get_plugin()

        # fetch the value so its stored in the cache
        p.cached_config

        # cache should now have finite size
        self.assertGreater(strax.config_cache_size_mb(), 0.0)

        # test if clearing cache works as expected
        strax.clear_config_caches()
        self.assertEqual(strax.config_cache_size_mb(), 0.0)

    # --- URL <-> AST conversion ---

    def test_url_to_ast(self):
        ast = URLConfigForTests.url_to_ast('take://json://{"a":1}?take=a')
        # kwargs are passed down to nested protocols that accept them (json takes **kwargs)
        self.assertEqual(ast, ("take", ("json", '{"a":1}', {"take": "a"}), {"take": "a"}))
        # ... and are dropped for protocols that do not accept them
        ast = URLConfigForTests.url_to_ast("take://only-a://x?a=1&take=b")
        # (take accepts **kwargs and keeps both)
        self.assertEqual(ast, ("take", ("only-a", "x", {"a": 1}), {"a": 1, "take": "b"}))

    def test_url_to_ast_unknown_protocol(self):
        self.assertEqual(URLConfigForTests.url_to_ast("unknown://abc"), (None, "unknown://abc", {}))
        self.assertEqual(URLConfigForTests.url_to_ast("abc"), (None, "abc", {}))
        with self.assertRaises(TypeError):
            URLConfigForTests.url_to_ast(42)

    def test_ast_to_url(self):
        # non-string arguments are wrapped in json://
        url = URLConfigForTests.ast_to_url("take", {"a": 1}, {"take": "a"})
        self.assertEqual(url, 'take://json://{"a": 1}?take=a')
        self.assertEqual(URLConfigForTests.ast_to_url("take", 1.5), "take://json://1.5")
        # nested AST tuples are supported
        url = URLConfigForTests.ast_to_url(("take", ("format", "{x}", {"x": 1}), {}))
        self.assertEqual(url, "take://format://{x}?x=1")
        # protocol None returns the argument as-is
        self.assertEqual(URLConfigForTests.ast_to_url(None, "abc"), "abc")
        with self.assertRaises(TypeError):
            URLConfigForTests.ast_to_url("take", object())

    def test_ast_roundtrip(self):
        url = 'take://json://{"a": {"b": 1}}?take=a&take=b'
        ast = URLConfigForTests.url_to_ast(url)
        self.assertEqual(URLConfigForTests.ast_to_url(*ast), url)
        self.assertEqual(URLConfigForTests.eval(url), 1)

    def test_are_equal(self):
        self.assertTrue(
            URLConfigForTests.are_equal(
                "format://{a}{b}?a=1&b=2",
                "format://{a}{b}?b=2&a=1",
            )
        )
        self.assertFalse(URLConfigForTests.are_equal("format://x?a=1", "format://x?a=2"))

    def test_url_kwargs(self):
        url = "format://x?b=2&a=1&c=1&c=2&d=text"
        path, kwargs = URLConfigForTests.split_url_kwargs(url)
        self.assertEqual(path, "format://x")
        self.assertEqual(kwargs, {"b": 2, "a": 1, "c": [1, 2], "d": "text"})
        self.assertEqual(URLConfigForTests.kwarg_from_url(url, "c"), [1, 2])
        self.assertIsNone(URLConfigForTests.kwarg_from_url(url, "missing"))
        # kwargs are sorted, new kwargs override existing ones
        self.assertEqual(
            URLConfigForTests.format_url_kwargs(url, d="new"),
            "format://x?a=1&b=2&c=1&c=2&d=new",
        )

    # --- Dry evaluation ---

    def test_dry_evaluation(self):
        """A dry evaluation outside of a plugin yields the same result as in the plugin."""
        plugin_url = "format://{run_id}?run_id=plugin.run_id"
        self.st.set_config({"test_config": plugin_url})
        correct_val = self.get_plugin().test_config

        dry_val1 = URLConfigForTests.evaluate_dry(f"format://{{run_id}}?run_id={RUN_ID}")
        dry_val2 = URLConfigForTests.evaluate_dry("format://{run_id}", run_id=RUN_ID)
        self.assertEqual(correct_val, dry_val1)
        self.assertEqual(correct_val, dry_val2)

        # dry-evaluation can not resolve plugin attributes
        with self.assertRaises(ValueError):
            URLConfigForTests.evaluate_dry(plugin_url)

    # --- Descriptions ---

    def test_protocol_descr(self):
        df = URLConfigForTests.protocol_descr()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("json://", df["name"].values)
        URLConfigForTests.print_protocols()

    def test_preprocessor_descr(self):
        df = URLConfigForTests.preprocessor_descr()
        self.assertEqual(len(df), len(URLConfigForTests._PREPROCESSORS))
        URLConfigForTests.print_preprocessors()
        URLConfigForTests.print_summary()
