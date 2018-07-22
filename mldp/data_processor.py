# Copyright (c) 2018 Juniper Overbeck
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from multiprocessing import Pool


def process_raw_data(data, reader=lambda x: x,
                     processor=lambda x: x):
    assert data is not None
    return processor(reader(data))


def pre_analysis_stage(data, wrapper_instructions):
    assert data is not None
    wrapper_dict = {'wrapper_data': data}
    for k in wrapper_instructions:
        wrapper_dict = k(wrapper_dict)
    return wrapper_dict


def model_testing(data,  test_schema, test_unit, analytics):
    with Pool() as p:
        results = p.map(test_unit, test_schema(data))
    return analytics(results)
