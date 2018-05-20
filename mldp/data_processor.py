def process_raw_data(data, reader=lambda x: x,
                     processor=lambda x: x):
    assert data is not None
    return processor(reader(data))


def wrapper_stage(data, wrapper_instructions):
    assert data is not None
    for k in wrapper_instructions:
        data = k(data)
    return data

def classify(data,  test_box, analytics):
    return None
