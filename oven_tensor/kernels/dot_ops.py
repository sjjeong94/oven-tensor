import oven.language as ol


def matmul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, m: int, n: int, k: int):
    block_size = ol.get_bdim_x()
    cCol = ol.get_bid_x()
    cRow = ol.get_bid_y()
    tCol = ol.get_tid_x()
    tRow = ol.get_tid_y()

    col = cCol * block_size + tCol
    row = cRow * block_size + tRow

    acc = 0.0
    for i in range(0, k, 1):
        a_offset = row * k + i
        b_offset = i * n + col

        a_val = ol.load(a_ptr, a_offset)
        b_val = ol.load(b_ptr, b_offset)
        acc = acc + a_val * b_val

    c_offset = row * n + col
    ol.store(acc, c_ptr, c_offset)
