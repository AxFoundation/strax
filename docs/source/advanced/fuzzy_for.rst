Fuzzy for is also working:
```python
st = straxen.contexts.xenonnt_online()
run_id, target = '022880', 'peak_basics'

st.is_stored(run_id, target)
-> True

st._plugin_class_registry[target].__version__ = 'does not excist'
st.is_stored(run_id, target)
-> False

st.context_config['fuzzy_for'] = (target,)
st.is_stored(run_id, target)
-> Was asked for 022880-peak_basics-kdfg6hmfkx returning /dali/lgrandi/rucio/xnt_022880/d3/b5/peak_basics-tsudklerox-metadata.json
-> True
```
