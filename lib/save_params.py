from flax.serialization import msgpack_serialize

def save_params(params, filename):
    serialized_params = msgpack_serialize(params)

    with open(filename, 'wb') as f:
        f.write(serialized_params)
