import multiprocessing
from tqdm import tqdm
import importlib

try:
    import mlcv_py
except:
    import sys
    sys.path.insert(0,r'C:\Users\vonst\Desktop\MLCV Project\python_bindings')
    import mlcv_py


funcs = {
    "test_border_encoding": mlcv_py.test_border_encoding,
    "test_adaptive_multicut_aware_encoding": mlcv_py.test_adaptive_multicut_aware_encoding,
    "test_ensemble_encoding": mlcv_py.test_ensemble_encoding,
    "make_mask_with_size": mlcv_py.make_mask_with_size,
    "encode_mask_with_size": mlcv_py.encode_mask_with_size
}

def run(data):
    name, args = data
    fn = funcs[name]
    return fn(*args)

def batch_execute(fn_name, inputs: list, cpu_count=None) -> list:
    """
    Executes the function `fn` on each tuple of `inputs` in parallel,
    using all available CPU cores, with a progress bar.
    
    Args:
        fn (callable): The function to execute.
        inputs (list of tuples): A list where each element is a tuple of arguments for `fn`.
    
    Returns:
        list: A list of results in the same order as the inputs.
    """
    if cpu_count is None:
        cpu_count = multiprocessing.cpu_count()
    
    results = []

    _inputs = [(fn_name, inp) for inp in inputs]

    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(pool.imap(run, _inputs), total=len(_inputs), desc="Processing"):
            results.append(result)
    return results