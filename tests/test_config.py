import json
import fsspec
from numpy.lib.arraysetops import isin
import strax


@strax.URLConfig.register('json')
def read_json(content, **kwargs):        
    return json.loads(content)

@strax.URLConfig.register('format')
def format_arg(arg, **kwargs):
    return arg.format(**kwargs)


class DummyPlugin(strax.Plugin):
    depends_on = ()

    # a simple lookup by key
    yay = strax.LookupConfig({321: 'yay!', 123: 'nay!'}, keys=('run_id',), )
    
     
    url_config = strax.URLConfig()
   
    func_config = strax.CallableConfig(func=lambda x,y: x+y, args=('run_id',))
    
def test_lookup_config():
    p = DummyPlugin()
    p.run_id = 321

    assert p.yay == 'yay!'

    p.run_id = 123

    assert p.yay == 'nay!'

def test_url_config():
    p = DummyPlugin()
    p.run_id = 321
    p.config = {
        'url_config': 'json://format://[{run_id}]?run_id=plugin.run_id',
    }
    assert p.url_config == [321]

def test_callable_config():
    p = DummyPlugin()
    p.run_id = 321
    p.config = {
        'func_config': 1,
    }
    assert p.func_config == 322
