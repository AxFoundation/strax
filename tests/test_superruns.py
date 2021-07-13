import os
import unittest
import shutil
import strax
import numpy as np
import tempfile
from strax.testutils import Records, PeaksWoByRunDefault
import datetime
import pytz
import json
from bson import json_util
import re


class TestSuperRuns(unittest.TestCase):

    def setUp(self, superrun_name='_superrun_test'):
        self.offset_between_subruns = 10
        self.superrun_name = superrun_name
        self.tempdir = tempfile.mkdtemp()
        self.context = strax.Context(storage=[strax.DataDirectory(self.tempdir,
                                                                  provide_run_metadata=True,
                                                                  readonly=False,
                                                                  deep_scan=True)],
                                     register=[Records, RecordsExtension],
                                     use_per_run_defaults=False)
        self.context.set_context_config({'write_superruns': True})
        
        logger = self.context.log
        logger.addFilter(lambda s: not re.match(".*Could not estimate run start and end time.*",
                                                s.getMessage()))

        self._create_subruns()
        self.context.define_run(self.superrun_name, data=self.subrun_ids)  # Define superrun

    def test_run_meta_data(self):
        """
        Check if superrun has the correct run start/end and livetime.
        """
        superrun_meta = self.context.run_metadata(self.superrun_name)
        subrun_meta = [self.context.run_metadata(r) for r in self.subrun_ids]

        assert superrun_meta['start'] == subrun_meta[0]['start']
        assert superrun_meta['end'] == subrun_meta[-1]['end']
        livetime = 0
        for meta in subrun_meta:
            time_delta = meta['end'] - meta['start']
            livetime += time_delta.total_seconds()*10**9
        assert superrun_meta['livetime'] == livetime

    def test_load_superruns(self):
        """
        Load superruns from already existing subruns. Does not write
        "new" data.
        """
        self.context.set_context_config({'write_superruns': False})
        subrun_data = self.context.get_array(self.subrun_ids,
                                             'records',
                                             progress_bar=False)
        superrun_data = self.context.get_array(self.superrun_name, 'records')
        assert np.all(subrun_data['time'] == superrun_data['time'])

        # Deactivate the lineage check for subruns:
        superrun_data = self.context.get_array(self.superrun_name,
                                               'records',
                                               _check_lineage_per_run_id=False
                                               )
        assert np.all(subrun_data['time'] == superrun_data['time'])
        assert not self.context.is_stored(self.superrun_name, 'records')

    def test_create_and_load_superruns(self):
        """
        Creates "new" superrun data from already existing data. Loads
        and compare data afterwards.
        """
        
        subrun_data = self.context.get_array(self.subrun_ids,
                                             'records',
                                             progress_bar=False)

        self.context.make(self.superrun_name,
                          'records',
                          _check_lineage_per_run_id=False
                          )
        superrun_data = self.context.get_array(self.superrun_name,
                                               'records',
                                               _check_lineage_per_run_id=False
                                               )

        assert self.context.is_stored(self.superrun_name, 'records')
        assert np.all(subrun_data['time'] == superrun_data['time'])

        # Load meta data and check if rechunking worked:
        chunks = self.context.get_meta(self.superrun_name, 'records')['chunks']
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk['run_id'] == self.superrun_name
        assert chunk['first_time'] == subrun_data['time'].min()
        assert chunk['last_endtime'] == np.max(strax.endtime(subrun_data))

    def test_select_runs(self):
        self.context.select_runs()
        self.context.make(self.superrun_name,
                          'records',
                          _check_lineage_per_run_id=False
                          )
        df = self.context.select_runs(available=('records',))
        assert self.superrun_name in df['name'].values

    def test_superrun_chunk_and_meta(self):
        """
        Superrun chunks and meta data should contain information about
        its constituent subruns.
        """
        self.context.make(self.superrun_name,
                          'records',
                          _check_lineage_per_run_id=False
                          )

        meta = self.context.get_meta(self.superrun_name, 'records')

        n_chunks = 0
        superrun_chunk = None
        for chunk in self.context.get_iter(self.superrun_name, 'records'):
            superrun_chunk = chunk
            n_chunks += 1

        assert len(meta['chunks']) == n_chunks == 1
        assert meta['chunks'][0]['subruns'] == superrun_chunk.subruns

        for subrun_id, start_and_end in superrun_chunk.subruns.items():
            rr = self.context.get_array(subrun_id, 'records')
            # Tests below only true for records as we have not rechunked yet.
            # After rechunking in general data start can be different from chunk start
            mes = f'Start time did not match for subrun: {subrun_id}'
            assert rr['time'].min() == start_and_end['start'], mes
            mes = f'End time did not match for subrun: {subrun_id}'
            assert np.max(strax.endtime(rr)) == start_and_end['end'], mes

    def test_merger_plugin_subruns(self):
        """
        Tests if merge plugins for subruns work. This test is needed to
        ensure that superruns do not interfer with merge plguns.
        (both are using underscores as identification).
        """
        rr = self.context.get_array(self.subrun_ids, ('records', 'records_extension'))
        p = self.context.get_single_plugin(self.subrun_ids[0], 'records_extension')
        assert np.all(rr['additional_field'] == p.config['some_additional_value'])
        
    def test_rechnunking_and_loading(self):
        """
        Tests rechunking and loading of superruns with multiple chunks.
        
        The test is required since it was possible to run into race conditions with 
        chunk.continuity_check in context.get_iter and transform_chunk_to_superrun_chunk 
        in storage.common.Saver.save_from.
        """
        self.context.set_config({'recs_per_chunk': 100000}) # Make chunks > 200 MB
        
        rr = self.context.get_array(self.subrun_ids, 'records')
        endtime = np.max(strax.endtime(rr))

        last_subrun_id = int(self.subrun_ids[-1])+1
        for run_id in range(last_subrun_id, last_subrun_id+2):
            self.context.set_config({'secret_time_offset': endtime + self.offset_between_subruns})
            rr = self.context.get_array(str(run_id), 'records')
            self._write_run_doc(self.context, run_id, 
                               self.now + datetime.timedelta(0, int(rr['time'].min())),
                               self.now + datetime.timedelta(0, int(np.max(strax.endtime(rr)))))
            endtime = np.max(strax.endtime(rr))
            self.subrun_ids.append(str(run_id))
            
        self.context.define_run('_superrun_test_rechunking', self.subrun_ids)
        self.context.make('_superrun_test_rechunking', 'records')
        
        rr_superrun = self.context.get_array('_superrun_test_rechunking', 'records')    
        rr_subruns = self.context.get_array(self.subrun_ids, 'records')    
        
        chunks = [chunk for chunk in self.context.get_iter('_superrun_test_rechunking', 'records')]
        assert len(chunks) > 1, 'Number of chunks should be larger 1. Has the default chunksize changed?'
        assert np.all(rr_superrun['time'] == rr_subruns['time'])

    def test_superrun_triggers_subrun_processing(self):
        """
        Tests if superrun processing can trigger subrun processing. Which it should.
        """
        self.context.register(PeaksWoByRunDefault)
        assert not self.context.is_stored(self.superrun_name, 'peaks')
        assert not self.context.is_stored(self.subrun_ids[0], 'peaks')

        self.context.make(self.superrun_name, 'peaks')
        assert self.context.is_stored(self.superrun_name, 'peaks')
        assert self.context.is_stored(self.subrun_ids[0], 'peaks')

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def _create_subruns(self, n_subruns=3):
        self.now = datetime.datetime.now()
        self.now.replace(tzinfo=pytz.utc)
        self.subrun_ids = [str(r) for r in range(n_subruns)]

        for run_id in self.subrun_ids:
            rr = self.context.get_array(run_id, 'records')
            time = np.min(rr['time'])
            endtime = np.max(strax.endtime(rr))

            self._write_run_doc(self.context,
                                run_id,
                                self.now + datetime.timedelta(0, int(time)),
                                self.now + datetime.timedelta(0, int(endtime)),
                                )

            self.context.set_config({'secret_time_offset': endtime + self.offset_between_subruns})
            assert self.context.is_stored(run_id, 'records')

    @staticmethod
    def _write_run_doc(context, run_id, time, endtime):
        """Function which writes a dummy run document.
        """
        run_doc = {'name': run_id, 'start': time, 'end': endtime}
        with open(context.storage[0]._run_meta_path(str(run_id)), 'w') as fp:
            json.dump(run_doc, fp, sort_keys=True, indent=4, default=json_util.default)


@strax.takes_config(
    strax.Option(
        name='some_additional_value',
        default=42,
        help="Some additional value for merger",
    )
)
class RecordsExtension(strax.Plugin):

    depends_on = 'records'
    provides = 'records_extension'
    dtype = strax.time_dt_fields + [(('Some additional field', 'additional_field'), np.int16)]

    def compute(self, records):

        res = np.zeros(len(records), self.dtype)
        res['time'] = records['time']
        res['length'] = records['length']
        res['dt'] = records['dt']
        res['additional_field'] = self.config['some_additional_value']
        return res
