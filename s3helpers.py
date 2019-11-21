import boto3
import contextlib
import h5py
import io
import os
import pickle
import tempfile
from keras.engine import topology


class s3Helpers():
    s3 = boto3.resource(
            's3',
            aws_access_key_id=os.environ['FEED_AWS_ACCESS_KEY'],
            aws_secret_access_key=os.environ['FEED_AWS_SECRET_ACCESS_KEY']
    )
    bucket = 'datasciencemodels.mgemi.com'

    def download_from_s3_local(self, file, loc):
        self.s3.Bucket(self.bucket).download_file(file, loc)

    def download_from_s3_io(self, file, func):
        with io.BytesIO() as data:
            self.s3.Bucket(self.bucket).download_fileobj(file, data)
            data.seek(0)    # move back to the beginning after writing
            return func(data)

    def load_h5_weights_from_s3(self, buf, model):
        file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        file_access_property_list.set_fapl_core(backing_store=False)
        file_access_property_list.set_file_image(buf)

        file_id_args = {
           'fapl': file_access_property_list,
           'flags': h5py.h5f.ACC_RDONLY,
           'name': next(tempfile._get_candidate_names()).encode(),
        }

        h5_file_args = {'backing_store': False, 'driver': 'core', 'mode': 'r'}

        with contextlib.closing(h5py.h5f.open(**file_id_args)) as file_id:
            with h5py.File(file_id, **h5_file_args) as h5_file:
                topology.load_weights_from_hdf5_group(h5_file, model.layers)
                return model

    def stream_h5_weights(self, file, model):
        with io.BytesIO() as data:
            self.s3.Bucket(self.bucket).download_fileobj(file, data)
            data.seek(0)
            self.load_h5_weights_from_s3(data.read(), model)

    def pickle_load_to_s3_io(self, ob, filename):
        out_buffer = io.BytesIO()
        with out_buffer as f:
            pickle.dump(ob, f)
            f.seek(0)
            self.s3.Object(self.bucket, filename).upload_fileobj(f) 
        out_buffer.close()


