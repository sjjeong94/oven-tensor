import oven.language as ol


def sigmoid(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Sigmoid activation function: 1 / (1 + exp(-x))"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.sigmoid(x_value)
    ol.store(y_value, y_ptr, idx)


def exp(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Exponential function: e^x"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.exp(x_value)
    ol.store(y_value, y_ptr, idx)


def sqrt(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Square root function"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.sqrt(x_value)
    ol.store(y_value, y_ptr, idx)


def abs(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Absolute value function"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.abs(x_value)
    ol.store(y_value, y_ptr, idx)


def ceil(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Ceiling function"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.ceil(x_value)
    ol.store(y_value, y_ptr, idx)


def floor(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Floor function"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.floor(x_value)
    ol.store(y_value, y_ptr, idx)


def rsqrt(x_ptr: ol.ptr, y_ptr: ol.ptr):
    """Reciprocal square root: 1/sqrt(x)"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.rsqrt(x_value)
    ol.store(y_value, y_ptr, idx)
