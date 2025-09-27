import oven.language as ol


def add(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    """Element-wise addition: a + b"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    a_value = ol.load(a_ptr, idx)
    b_value = ol.load(b_ptr, idx)
    c_value = a_value + b_value
    ol.store(c_value, c_ptr, idx)


def mul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    """Element-wise multiplication: a * b"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    a_value = ol.load(a_ptr, idx)
    b_value = ol.load(b_ptr, idx)
    c_value = a_value * b_value
    ol.store(c_value, c_ptr, idx)


def sub(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    """Element-wise subtraction: a - b"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    a_value = ol.load(a_ptr, idx)
    b_value = ol.load(b_ptr, idx)
    c_value = a_value - b_value
    ol.store(c_value, c_ptr, idx)


def div(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    """Element-wise division: a / b"""
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    a_value = ol.load(a_ptr, idx)
    b_value = ol.load(b_ptr, idx)
    c_value = a_value / b_value
    ol.store(c_value, c_ptr, idx)
