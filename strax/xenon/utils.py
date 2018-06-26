import os
import urllib.request

import strax

cache_folder = './resource_cache'


# Placeholder for resource management system in the future?
def get_resource(x, binary=False):
    """Return contents of file or URL x
    :param binary: Resource is binary. Return bytes instead of a string.
    """
    if '://' in x:
        # Web resource
        cache_f = os.path.join(cache_folder,
                               strax.utils.deterministic_hash(x))
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        if not os.path.exists(cache_f):
            with open(cache_f, mode='wb' if binary else 'w') as f:
                y = urllib.request.urlopen(x).read()
                if not binary:
                    y = y.decode()
                f.write(y)
        return get_resource(cache_f, binary=binary)
    else:
        # File resource
        with open(x, mode='rb' if binary else 'r') as f:
            return f.read()
