import jax
import jax.random as rand

device_cpu = None

def do_on_cpu(f):
    global device_cpu
    if device_cpu is None:
        device_cpu = jax.devices('cpu')[0]

    def inner(*args, **kwargs):
        with jax.default_device(device_cpu):
            return f(*args, **kwargs)
    return inner

seed2key = do_on_cpu(rand.PRNGKey)
seed2key.__doc__ = '''Same as `jax.random.PRNGKey`, but always produces the result on CPU.'''
