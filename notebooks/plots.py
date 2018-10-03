"""Plotting helper functions for the online demo notebook
The code in show_time_range is mostly copy-pasted from 
the visualization notebook.
"""
import time
import numpy as np
import matplotlib.pyplot as plt


def event_scatter(df, sleep_factor=2, max_delay=60, time_cut=False, s=5, update=False):
    max_delay *= 1e9

    # Select recent events
    if time_cut:
        now = time.time()
        df['delay'] = int(now) * int(1e9) - df['time']
        df = df[df['delay'] < int(max_delay)]
        s = 200 * (max_delay - df.delay) / max_delay
    else:
        s = s
        if len(df) > 10000:
            df = df.sample(10000)

    # Update the plot
    plt.gca().clear()

    plt.scatter(np.nan_to_num(df.cs1),
                np.nan_to_num(df.cs2),
                c=df.s1_area_fraction_top,
                vmin=0, vmax=0.3,
                s=s,
                cmap=plt.cm.jet,
                marker='.', edgecolors='none')
    if not update:
        plt.colorbar(label="S1 area fraction top", extend='max')
    plt.xlabel('cS1 (PE)')
    plt.ylabel('cS2 (PE)')
    plt.xscale('symlog')
    plt.yscale('log')
    plt.ylim(1e1, 1e7)
    plt.xlim(-0.1, 1e6)
    if time_cut:
        plt.title(time.strftime('%H:%M:%S'))

    plt.gcf().canvas.draw()

    if time_cut:
        # Sleep for some fraction of the time it took to make the plot
        time.sleep(sleep_factor * (time.time() - now))



def show_time_range(st, run_id, t0, dt=10):
    from functools import partial

    import numpy as np
    import pandas as pd

    import holoviews as hv
    from holoviews.operation.datashader import datashade, dynspread
    hv.extension('bokeh')

    import strax

    import gc
    # Somebody thought it was a good idea to call gc.collect explicitly somewhere in holoviews
    # This makes dynamic PMT maps super slow
    # Until I trace the offender:
    gc.collect = lambda *args, **kwargs: None

    # Custom wheel zoom tool that only zooms in time
    from bokeh.models import WheelZoomTool
    time_zoom = WheelZoomTool(dimensions='width')

    # Get ADC->pe multiplicative conversion factor
    from pax.configuration import load_configuration
    from pax.dsputils import adc_to_pe
    pax_config = load_configuration('XENON1T')["DEFAULT"]
    to_pe = np.array([adc_to_pe(pax_config, ch)
                      for ch in range(pax_config['n_channels'])])

    tpc_r = pax_config['tpc_radius']

    # Get locations of PMTs
    r = []
    for q in pax_config['pmts']:
        r.append(dict(x=q['position']['x'],
                      y=q['position']['y'],
                      i=q['pmt_position'],
                      array=q.get('array', 'other')))
    f = 1.08
    pmt_locs = pd.DataFrame(r)

    records = st.get_array(run_id, 'raw_records', time_range=(t0, t0 + int(1e10)))

    # TOOD: don't reprocess, just load...
    hits = strax.find_hits(records)
    peaks = strax.find_peaks(hits, to_pe, gap_threshold=300, min_hits=3,
                             result_dtype=strax.peak_dtype(n_channels=260))
    strax.sum_waveform(peaks, records, to_pe)
    # Integral in pe
    areas = records['data'].sum(axis=1) * to_pe[records['channel']]

    def normalize_time(t):
        return (t - records[0]['time']) / 1e9

    # Create dataframe with record metadata
    df = pd.DataFrame(dict(area=areas,
                           time=normalize_time(records['time']),
                           channel=records['channel']))

    # Convert to holoviews Points
    points = hv.Points(df,
                       kdims=[hv.Dimension('time', label='Time', unit='sec'),
                              hv.Dimension('channel', label='PMT number', range=(0, 260))],
                       vdims=[hv.Dimension('area', label='Area', unit='pe',
                                           # range=(0, 1000)
                                           )])

    def pmt_map(t_0, t_1, array='top', **kwargs):
        # Compute the PMT pattern (fast)
        ps = points[(t_0 <= points['time'])
                    & (points['time'] < t_1)]
        areas = np.bincount(ps['channel'],
                            weights=ps['area'],
                            minlength=len(pmt_locs))

        # Which PMTs should we include?
        pmt_mask = pmt_locs['array'] == array
        d = pmt_locs[pmt_mask].copy()
        d['area'] = areas[pmt_mask]

        # Convert to holoviews points
        d = hv.Dataset(d,
                       kdims=[hv.Dimension('x', unit='cm', range=(-tpc_r * f, tpc_r * f)),
                              hv.Dimension('y', unit='cm', range=(-tpc_r * f, tpc_r * f)),
                              hv.Dimension('i', label='PMT number'),
                              hv.Dimension('area',
                                           label='Area',
                                           unit='PE')])

        return d.to(hv.Points,
                    vdims=['area', 'i'],
                    group='PMTPattern',
                    label=array.capitalize(),
                    **kwargs).opts(
            plot=dict(color_index=2,
                      tools=['hover'],
                      show_grid=False),
            style=dict(size=17,
                       cmap='magma'))

    def pmt_map_range(x_range, array='top', **kwargs):
        # For use in dynamicmap with streams
        if x_range is None:
            x_range = (0, 0)
        return pmt_map(x_range[0], x_range[1], array=array, **kwargs)

    xrange_stream = hv.streams.RangeX(source=points)

    # TODO: weigh by area

    def channel_map():
        return dynspread(datashade(points,
                                   y_range=(0, 260),
                                   streams=[xrange_stream])).opts(
            plot=dict(width=600,
                      tools=[time_zoom, 'xpan'],
                      default_tools=['save', 'pan', 'box_zoom', 'save', 'reset'],
                      show_grid=False))

    def plot_peak(p):
        # It's better to plot amplitude /time than per bin, since
        # sampling times are now variable
        y = p['data'][:p['length']] / p['dt']
        t_edges = np.arange(p['length'] + 1, dtype=np.int64)
        t_edges = t_edges * p['dt'] + p['time']
        t_edges = normalize_time(t_edges)

        # Correct step plotting from Knut
        t_ = np.zeros(2 * len(y))
        y_ = np.zeros(2 * len(y))
        t_[0::2] = t_edges[0:-1]
        t_[1::2] = t_edges[1::]
        y_[0::2] = y
        y_[1::2] = y

        c = hv.Curve(dict(time=t_, amplitude=y_),
                     kdims=points.kdims[0],
                     vdims=hv.Dimension('amplitude', label='Amplitude', unit='PE/ns'),
                     group='PeakSumWaveform')
        return c.opts(plot=dict(  # interpolation='steps-mid',
            # default_tools=['save', 'pan', 'box_zoom', 'save', 'reset'],
            # tools=[time_zoom, 'xpan'],
            width=600,
            shared_axes=False,
            show_grid=True),
            style=dict(color='b')
            # norm=dict(framewise=True)
        )

    def peaks_in(t_0, t_1):
        return peaks[(normalize_time(peaks['time'] + peaks['length'] * peaks['dt']) > t_0)
                     & (normalize_time(peaks['time']) < t_1)]

    def plot_peaks(t_0, t_1, n_max=10):
        # Find peaks in this range
        ps = peaks_in(t_0, t_1)
        # Show only the largest n_max peaks
        if len(ps) > n_max:
            areas = ps['area']
            max_area = np.sort(areas)[-n_max]
            ps = ps[areas >= max_area]


        return hv.Overlay(items=[plot_peak(p) for p in ps])

    def plot_peak_range(x_range, **kwargs):
        # For use in dynamicmap with streams
        if x_range is None:
            x_range = (0, 10)
        return plot_peaks(x_range[0], x_range[1], **kwargs)


    top_map = hv.DynamicMap(partial(pmt_map_range, array='top'), streams=[xrange_stream])
    bot_map = hv.DynamicMap(partial(pmt_map_range, array='bottom'), streams=[xrange_stream])
    waveform = hv.DynamicMap(plot_peak_range, streams=[xrange_stream])
    layout = waveform + top_map + channel_map() + bot_map
    return layout.cols(2)