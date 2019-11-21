import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import requests
import scipy as sp

from sklearn.preprocessing import LabelEncoder, StandardScaler
from spotlight.interactions import Interactions

from svr.browse_model.browse_model_helpers import ModelHelpers
from svr.dbhelpers import DBHelpers
from svr.utils.s3helpers import s3Helpers

from svr.utils.mgemi import ft_master_orders_name

class ModelPipeline():

    def __init__(self):
        self.dbHelpers = DBHelpers()
        self.s3Helpers = s3Helpers()
        self.ModelHelpers = ModelHelpers()

    send_views_query = """
        with single_email as (
        select uuid, count(distinct email) as count_emails
        from datascience.email_uuids
        group by 1
        having count(distinct email) = 1
        )
        select eu.email,
        extract (epoch from '{update_time_dt}'::timestamp - to_timestamp(max(cpv.timestamp)/1000))  as time_since_browse,
        count(case when split_part(split_part(query_string,'cgid=',2),'&',1) <> '' then cpv.timestamp else NULL end) as cgid_count,
        count(case when SPLIT_PART(SPLIT_PART(SPLIT_PART(cpv.query_string,'pid=',2),'&',1),'?',1) <> '' then cpv.timestamp else NULL end) as pdp_count,
        count(distinct cpv.timestamp)
        from commersive.clickstream_page_views cpv
        join datascience.email_uuids eu on cpv.user_uuid = eu.uuid
        join single_email se on eu.uuid = se.uuid
        where
        cpv.timestamp >= 1000*EXTRACT(epoch FROM '{update_time_dt}'::TIMESTAMP at time zone 'UTC' - interval '5 days + 2 hours')::BIGINT
        and cpv.timestamp < 1000*EXTRACT(epoch FROM '{update_time_dt}'::TIMESTAMP at time zone 'UTC' - interval '2 hours')::BIGINT
        AND cpv.host in ('mgemi.com', 'www.mgemi.com', 'production-veloce-rcw.demandware.net')
        group by 1
        having max(timestamp) >= 1000*EXTRACT(epoch FROM '{last_run_dt}'::TIMESTAMP at time zone 'UTC' - interval '9  hours')::BIGINT
        order by 1,2,3
        ;"""
    seq_views_query = """
         select eu.email, cpv.session_id,
         to_timestamp(timestamp/1000) as activity_dt,
         date_trunc('day',to_timestamp(timestamp/1000)) as day,
         case when split_part(split_part(query_string,'cgid=',2),'&',1) <> '' then split_part(split_part(query_string,'cgid=',2),'&',1)
         else SPLIT_PART(SPLIT_PART(SPLIT_PART(cpv.query_string,'pid=',2),'&',1),'?',1)
         end as activity,
         case when split_part(split_part(query_string,'cgid=',2),'&',1) <> '' then 'cgid' else 'pid' end as activity_type,
         regexp_replace(SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(url,'/color/',2),'/',1),'?',1),'&',1),'http',1), '[^0-9]+','') as color_code
         from commersive.clickstream_page_views cpv
         join datascience.email_uuids eu on cpv.user_uuid = eu.uuid
         where (SPLIT_PART(SPLIT_PART(SPLIT_PART(cpv.query_string,'pid=',2),'&',1),'?',1) <> ''
          or split_part(split_part(query_string,'cgid=',2),'&',1) <> '')
          and cpv.timestamp >= 1000*EXTRACT(epoch FROM '{}'::TIMESTAMP at time zone 'UTC' - interval '9 hours')::BIGINT
          and cpv.timestamp < 1000*EXTRACT(epoch FROM '{}'::TIMESTAMP at time zone 'UTC' - interval '2 hours')::BIGINT
          AND cpv.host in ('mgemi.com', 'www.mgemi.com', 'production-veloce-rcw.demandware.net')
         order by 1,2

         ;"""

    user_query = """
         SELECT
         lower(billing_email) as email,
         COUNT(distinct order_id) as lifetime_orders,
         DATE_PART('days', '{}' - MAX(a.order_date)) as days_since_last_order
           FROM {} a
           GROUP BY 1
         ;"""

    def send_model_pipeline(self, update_time, last_run, model_files):
        """
        Builds features for send model
        """
        scaler = self.s3Helpers.download_from_s3_io(
             model_files['send_scaler']['s3loc'],
             pickle.load)

        df = self.dbHelpers.query_postgres(
         self.send_views_query.format(update_time_dt=update_time, last_run_dt=last_run)
         )

        ft_name = ft_master_orders_name(self.dbHelpers)
        user_df = self.dbHelpers.query_postgres(self.user_query.format(update_time,ft_name))

        X = df.merge(user_df[['email','lifetime_orders','days_since_last_order']],
         on='email', how='left')
        X.lifetime_orders.fillna(0, inplace=True)
        X.days_since_last_order.fillna(100, inplace=True)
        users = X.pop('email')
        Xs = scaler.transform(X)
        return users, Xs

    def sequence_model_pipeline(self, users, update_time, last_run, model_files):
        """
        Builds index arrays and sequences for sequence model,
        filtered on users over send threshold
        """
        df = self.dbHelpers.query_postgres(
         self.seq_views_query.format(last_run, update_time)
         )

        model_info = self.s3Helpers.download_from_s3_io(
         model_files['model_info']['s3loc'],
         pickle.load)
        le_cat_pid = model_info['sqmodel_le_cat_pid']
        df = df[df.email.isin(users) & df.activity.isin(le_cat_pid.classes_)] # same pids as in training df.activity.isin(le_cat_pid.classes_) &
        df_pid = df[df.activity_type == 'pid']
        df_cat = df[df.activity_type == 'cgid']

        le_user = LabelEncoder()
        user_idx = le_user.fit_transform(df['email'])   # fit on everything

        user_idx_pid = le_user.transform(df_pid['email'])
        pid_idx = le_cat_pid.transform(df_pid['activity'])

        user_idx_cat = le_user.transform(df_cat['email'])
        cat_idx = le_cat_pid.transform(df_cat['activity'])

        # add 1 to cat_pid_idx so there are no zeroes - function won't take them
        pid_implicit_interactions = Interactions(
         user_idx_pid, pid_idx + 1, timestamps = df_pid['activity_dt']
         )
        pid_sequential_interactions = pid_implicit_interactions.to_sequence(
         max_sequence_length=10,step_size=10)

        cat_implicit_interactions = Interactions(
         user_idx_cat, cat_idx + 1, timestamps = df_cat['activity_dt']
         )
        cat_sequential_interactions = cat_implicit_interactions.to_sequence(
         max_sequence_length=10,step_size=10)
        return pid_sequential_interactions, cat_sequential_interactions,\
         le_user, df_pid[['email','activity','color_code']], df_cat[['email','activity','color_code']],set(df.email.unique())




    def main():
        startdt = '2018-07-20'
        enddt = '2018-07-21'
        bucket = 'datasciencemodels.mgemi.com'

