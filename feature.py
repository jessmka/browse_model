from datetime import datetime
import hashlib
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import requests
import scipy as sp
from svr.utils.s3helpers import s3Helpers

from svr.browse_model.models.sequence_model import SequenceModel
from svr.browse_model.models.send_model import SendModel
from svr.browse_model.browse_model_helpers import ModelHelpers
from svr.browse_model.pipelines import ModelPipeline
from svr.dbhelpers import DBHelpers
from svr.abandon_browse_cache_tables import BrowseHelpers

class BrowseFeature():

    FEATURE_NAME = 'browse_model'

    insert_send_flag = """
        UPDATE datascience.send_model_output
        SET send_flag = TRUE
        WHERE email = '{}'
        and timestamp = '{}';
        """
    last_run_insert = """
        insert into datascience.abandon_browse_last_run
        values ('{}','{}')
        """
    DEPLOYMENT_THRESHOLD = 0.9
    categories_to_send = ['moccasins', 'pumps','womens_pumps','mens-sneakers']

    def do_models(self, last_run, update_time, n=None):
        mh = ModelHelpers()
        mp = ModelPipeline()
        sm = SendModel()
        sq = SequenceModel()
        bh = BrowseHelpers()
        model_files = mh.model_files()

        users, X_scaled = mp.send_model_pipeline(update_time, last_run, model_files)
        send_pred = sm.predict_model(X_scaled, model_files)
        send_df = mh.send_model_output_to_db(users, send_pred, update_time) #this outputs the df
        send_users = mh.send_threshold_and_random(send_df, mh.SEND_THRESHOLD, mh.FRACTION_SEND, update_time)
        pid_seq_data, cat_seq_data, le_user, pid_df, cat_df, user_array = mp.sequence_model_pipeline(
            send_users, update_time, last_run, model_files
            )
        pid_pred_df = pd.DataFrame(sq.predict_model(pid_seq_data, 'pid', le_user, model_files))
        cat_pred_df = pd.DataFrame(sq.predict_model(cat_seq_data, 'cgid', le_user, model_files))
        color_df = mh.color_codes_to_list_df(pid_df)
        pid_pred_df = pid_pred_df.merge(color_df, on=['email','activity'], how='left')
        pid_pred_df['activity_type'] = 'pid'
        cat_pred_df['activity_type'] = 'cgid'
        cat_pred_df['color_code'] = -1

        cat_pred_df = cat_pred_df[cat_pred_df['activity'].isin(self.categories_to_send)]

        mh.seq_model_output_to_db(pid_pred_df, cat_pred_df, update_time)
        
        cat_df_f = mh.category_map_to_product(cat_pred_df)
        
        # Check inventory
        out_df = pd.concat([pid_pred_df, cat_df_f], axis = 0) #commenting out right now as not using categories
        out_df = mh.check_inventory_on_pred_df(out_df)
        out_df = pd.concat([out_df, cat_pred_df], axis=0)
        cols = ['activity','activity_type','email','sksz_color_code','pred']
        out_df = mh.get_max_row_dedupe(out_df, 'email','pred')

        #  make payload
        product_variant_df = mh.get_variant_fields(out_df)
        out_df = out_df.merge(product_variant_df, left_on=['activity','sksz_color_code'],
            right_on = ['activity','pv_color_code'],how='left')
        out_df.rename(columns={'activity':'pid','color_code':'color_list', 'sksz_color_code':'color_code'}, inplace=True) #renaming to pid for dict function
        pd.set_option('display.max_columns',None)
        DBHelpers().execute_query(self.last_run_insert.format(update_time, self.FEATURE_NAME))
        max_hash_sample = (2**224)*self.DEPLOYMENT_THRESHOLD
        if n is None:
            n = out_df.shape[0]
        j = 0
        for i, row in out_df.iterrows():
            hash_object = hashlib.sha224(str.encode(row.email)).hexdigest()
            payload = dict(email=row.email, eventName = 'browse-model',
                dataFields=dict())
            if row.activity_type=='pid':
                payload['dataFields']['products'] = [bh.anniv_product_dict(row)]
            if row.activity_type=='cgid':
                payload['dataFields']['category'] = row.pid
                
            if int(hash_object,16) <= max_hash_sample and j < int(n):
                try:
                    requests.post('https://api.iterable.com/api/events/track',
                                json=payload, params=dict(api_key=bh.API_KEY))
                    DBHelpers().execute_query(self.insert_send_flag.format(row.email, update_time))
                    j += 1
                except requests.exceptions.RequestException as e:
                        logging.exception(e)