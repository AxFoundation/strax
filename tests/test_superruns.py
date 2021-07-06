import os
import unittest
import shutil
import strax
import numpy as np
import tempfile
from strax.testutils import Records
import datetime
import pytz
import json
from bson import json_util


class TestSuperRuns(unittest.TestCase):

    def setUp(self, superrun_name='_superrun_test'):
        self.superrun_name = superrun_name
        self.tempdir = tempfile.mkdtemp()
        self.context = strax.Context(storage=[strax.DataDirectory(self.tempdir,
                                                                  provide_run_metadata=True,
                                                                  readonly=False,
                                                                  deep_scan=True)],
                                     register=[Records],
                                     use_per_run_defaults=False)
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
        self.context.set_context_config({'storage_converter': True})
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
        self.context.set_context_config({'storage_converter': True})
        self.context.make(self.superrun_name,
                          'records',
                          _check_lineage_per_run_id=False
                          )
        df = self.context.select_runs(available=('records',))
        assert self.superrun_name in df['name'].values

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def _create_subruns(self, n_subruns=3, offset_between_subruns=2):
        now = datetime.datetime.now()
        now.replace(tzinfo=pytz.utc)
        self.subrun_ids = [str(r) for r in range(n_subruns)]

        for run_id in self.subrun_ids:
            rr = self.context.get_array(run_id, 'records')
            time = np.min(rr['time'])
            endtime = np.max(strax.endtime(rr))

            self._write_run_doc(self.context,
                                run_id,
                                now + datetime.timedelta(0, int(time)),
                                now + datetime.timedelta(0, int(endtime)),
                                )

            self.context.set_config({'secret_time_offset': endtime + offset_between_subruns})
            assert self.context.is_stored(run_id, 'records')

    @staticmethod
    def _write_run_doc(context, run_id, time, endtime):
        """Function which writes a dummy run document.
        """
        run_doc = {'name': run_id, 'start': time, 'end': endtime}
        with open(context.storage[0]._run_meta_path(str(run_id)), 'w') as fp:
            json.dump(run_doc, fp, sort_keys=True, indent=4, default=json_util.default)
