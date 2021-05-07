from collections.abc import Iterable
from abc import ABC, abstractmethod
from collections import defaultdict
from intervaltree import Interval, IntervalTree
import typing as ty
from dataclasses import dataclass
import time


@dataclass
class Version:
    value: typing.Any = float("nan")
    priority: int = 0
    created: float = float("nan")
    
    def __hash__(self):
        return hash(self.value)

def sorted_versions(intervals):
    return [iv.data for iv in sorted(intervals, key=lambda iv:iv.data.priority)]


def sorted_versions_tree(intervals):
    tree = IntervalTree(intervals)
    tree.split_overlaps()
    bounds = list(tree.boundary_table)
    versions = set()
    for s,e in zip(bounds[:-1], bounds[1:]):
        overlaps  = tree.overlap(s,e)
        iv = Interval(s,e, sorted_versions(overlaps))
        versions.add(iv)
    return IntervalTree(versions)

def IntervalEngine(ABC):

    @abstractmethod
    def add(self, interval: Interval)->None:
        pass
    
    @abstractmethod
    def overlap(self, begin: int, end:int=None)->ty.Set[Interval]:
        pass
    
    @abstractmethod
    def at(self, index:int)->ty.Set[Interval]:
        pass
    
    @abstractmethod
    def begin(self)->int:
        pass
    
    @abstractmethod
    def end(self)->int:
        pass
    
class VersionedValidityIntervals:
    ENGINE_CLASS = IntervalTree
    
    def __init__(self, engine=None, default = float("nan")):
        if isinstance(engine, self.ENGINE_CLASS):
            self._engine = engine
        elif isinstance(engine, Iterable):
            self._engine = self.ENGINE_CLASS(engine)
        else:
            self._engine = None
        
        self.default = default
    
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
        elif isinstance(key, tuple):
            if len(key)==2:
                start, stop = key
                step = None
            if len(key)==3:
                start, stop, step = key
            else:
                raise ValueError("Setting intervals with tuple must be  \
                            of form (start, end) or (start, end, step)")
        else:
            raise TypeError("Wrong type. Setting intervals can only be done using a \
                            slice or tuple of (start, end) or (start, end, step)")
        if start is None:
            start = self.start
        if stop is None:
            stop = self.end
        if step is None:
            self.add_interval(start, stop, value)
        else:
            self.add_intervals(start,stop,step,value)
            
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.value(key)
        elif isinstance(key, Iterable):
            return self.values(key)
        elif isinstance(key, slice):
            if key.step is None:
                return self.cut(key.start, key.stop)
            else:
                return self.values(range(key.start, key.stop, key.step))
    
    def get_engine(self):
        if self._engine is None:
            self._engine = self.ENGINE_CLASS()
        return self._engine
    
    @property
    def start(self):
        return self.get_engine().begin()
    
    @property
    def end(self):
        return self.get_engine().end()
    
    def add_interval(self, start, end, value):
        overlaps = self.get_engine().overlap(start, end)
        if overlaps:
            priority = max([iv.data.priority for iv in overlaps]) + 1
        else:
            priority = 0
        data = Version(priority=priority, value=value, created=time.time())
        interval = Interval(start, end, data)
        self.get_engine().add(interval)
        
    def add_interval_range(self, start, end, step, values):
        if not isinstance(values, Iterable):
            values = [values]
        bounds = zip(range(start, end-step), range(start+step, end))
        if len(bounds)!=len(values):
            raise ValueError(f"Values wrong shape. Interval range has length {len(bounds)},\
                             values given is of length {len(values)}")
        for bound,v in zip(bounds, values):
            self.add_interval(*bound, v)
            
    @staticmethod
    def data_reducer(current_reduced_data, new_data):
        if current_reduced_data.priority>new_data.priority:
            return current_reduced_data
        else:
            return new_data
        
    def versions(self, index):
        intervals = self.get_engine().at(index)
        return sorted_versions(intervals)
    
    def overlap_versions(self, start, end=None):
        if end is None:
            end = start.end
        versions = sorted_versions_tree(self.get_engine().overlap(start, end))
        return versions
    
    def overlap(self, start=None, end=None,  max_version=None):
        if start is None:
            start = self.start
        elif end is None:
            start, end = start.begin, start.end
        else:
            raise ValueError("Must provide start or end boundaries.")
        hits = self.get_engine().overlap(start, end-1)
        tree = sorted_versions_tree(hits)      
        overlaps = set(Interval(iv.begin, iv.end, iv.data[slice(max_version)][-1])
                       for iv in tree if iv.data)
        return overlaps
    
    def cut(self, start=None, end=None, max_version=None):
        overlaps = self.overlap(start,end,max_version)
        return type(self)(overlaps, default=self.default)
    
    def value(self, index, max_version=None):
        versions = self.versions(index)
        try:
            val = versions[slice(max_version)][-1].value
            return val
        except IndexError:
            if self.default is None:
                raise IndexError("Value not defined for given index and not fill value given.")
            self.add_interval(index, index+1, self.default)
            return self.default

    @property
    def last_value(self):
        return self.value(self.end-1)
    
    def values(self, indices, max_version=None):
        return [self.value(i, max_version=max_version) for i in indices]
        
    def to_dataframe(self, start=None, end=None):
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        tree = self.overlap_versions(start, end)
        records = []
        for iv in tree:
            record = {f"v{i}": v.value for i,v in enumerate(iv.data)}
            record["interval"] = pd.Interval(iv.begin, iv.end, closed="left")
            records.append(record)
        df = pd.DataFrame.from_records(records).set_index("interval")

        return df
        
    def __repr__(self):
        return f"{self.__class__.__name__}(start={self.start} end={self.end} default={self.default})"
    

class VersionedValue:
    def __init__(self, default=None):
        self.default=default
        
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name + "_intervals"
        
    def __get__(self, obj, objtype=None):
        if not hasattr(obj, self.private_name):
            intervals = VersionedValidityIntervals(default=self.default)
            setattr(obj, self.private_name, intervals)
        return getattr(obj, self.private_name)
