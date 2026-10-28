"""Model-free tests for the development conversion boundary."""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import prepare_soma_v12 as prepare
import convert_motion_to_gguf as converter


class Preparation(unittest.TestCase):
    def test_license_and_missing_input(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'input'; output=Path(d)/'out.gguf'
            with self.assertRaisesRegex(ValueError,'accept-licenses'):
                prepare.prepare(source,output,False,False)
            source.mkdir();(source/'LICENSE').write_text('different terms')
            with self.assertRaisesRegex(ValueError,'checksum mismatch: LICENSE'):
                prepare.prepare(source,output,False,True)
            self.assertFalse(output.exists())

    def test_float64_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'stat.npy'
            header=b"{'descr': '<f8', 'fortran_order': False, 'shape': (2,)}\n"
            path.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<dd',.25,1.5))
            tensor=converter.read_npy(path,'stats.test')
            output=Path(d)/'raw'
            with output.open('wb') as stream:converter.copy_range(stream,tensor)
            self.assertEqual(output.read_bytes(),struct.pack('<ff',.25,1.5))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'truncated'):converter.read_npy(path,'bad')

    def test_safetensors_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.safetensors'
            path.write_bytes(struct.pack('<Q',1000000000))
            with self.assertRaisesRegex(ValueError,'header too large'):converter.read_safetensors(path)


if __name__=='__main__':unittest.main()
