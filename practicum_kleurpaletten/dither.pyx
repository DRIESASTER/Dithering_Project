import numpy as np
cimport numpy as np
np.import_array()
cimport cython
import tqdm


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def transform_image_to_indices_diffusion_dithering(
        np.ndarray[np.uint8_t,ndim=3] img,
        np.ndarray[np.uint8_t,ndim=2] color_table,
        np.ndarray[np.float32_t, ndim=2] dither_matrix,
        int anchor_col):
    # print(color_table)
    """
    Transforms a full-color image to the grid of indices where every full-color pixel is mapped
    onto one index. This process applies error diffusion using the supplied `dither_matrix`.
    The algorithm diffuses errors either:
        - to the right on the same scanline,
        - or downward (both left and right are possible).
    The `anchor_col` defines which column of the `dither_matrix` contains the anchor point.
    """
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] color_table_indices = np.zeros((rows, cols), dtype=np.uint8)

    cdef int dither_rows = dither_matrix.shape[0]
    cdef int dither_cols = dither_matrix.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] error
    cdef np.ndarray[np.float32_t, ndim=1] new_val
    cdef int new_color_index = 0
    # cdef np.ndarray[np.float32_t, ndim=1] new_color
    cdef int row = -1
    cdef int col = -1
    cdef int dither_row = -1
    cdef int dither_col = -1
    cdef int i = -1
    # cdef np.ndarray[np.float32_t, ndim=1] current_color = np.zeros(3, dtype=np.float32)
    cdef int anchor = anchor_col


    # TODO: implement diffusion dithering
    for _ in range(rows):
        row+= 1
        col = -1
        for _ in range(cols):
            col+= 1
            # current_color = img[row,col].astype(np.float32)
            new_color_index = find_closest_color(img[row,col], color_table)
            color_table_indices[row,col] = new_color_index
            # new_color = np.array(color_table[new_color_index], dtype=np.float32)
            error_r = <int> (img[row,col,0]) - <int> (color_table[new_color_index,0])
            error_g = <int> (img[row,col,1]) - <int> (color_table[new_color_index,1])
            error_b = <int> (img[row,col,2]) - <int> (color_table[new_color_index,2])
            # error = np.array(img[row,col], dtype=np.float32) - np.array(color_table[new_color_index], dtype=np.float32)
            dither_row = -1
            for _ in range(dither_rows):
                dither_row+= 1
                dither_col= -1
                for _ in range(dither_cols):
                    dither_col+= 1
                    if col + dither_col - anchor >= cols:
                        continue
                    if row + dither_row >= rows:
                        continue
                    # new_val = img[row+dither_row, col+dither_col - anchor].astype(np.float32) + error * dither_matrix[dither_row, dither_col]
                    new_r = img[row+dither_row, col+dither_col - anchor,0] + error_r * dither_matrix[dither_row, dither_col]
                    new_g = img[row+dither_row, col+dither_col - anchor,1] + error_g * dither_matrix[dither_row, dither_col]
                    new_b = img[row+dither_row, col+dither_col - anchor,2] + error_b * dither_matrix[dither_row, dither_col]

                    # new_r = <int> (img[row+dither_row, col+dither_col - anchor,0]) + error_r * <int> (dither_matrix[dither_row, dither_col])
                    # new_g = <int> (img[row+dither_row, col+dither_col - anchor,1]) + error_g * <int> (dither_matrix[dither_row, dither_col])
                    # new_b = <int> (img[row+dither_row, col+dither_col - anchor,2]) + error_b * <int> (dither_matrix[dither_row, dither_col])
                    if new_r < 0:
                        new_r = 0
                    elif new_r > 255:
                        new_r = 255
                    if new_g < 0:
                        new_g = 0
                    elif new_g > 255:
                        new_g = 255
                    if new_b < 0:
                        new_b = 0
                    elif new_b > 255:
                        new_b = 255
                    # new_val = img[row+dither_row, col+dither_col - anchor].astype(np.float32) + error * dither_matrix[dither_row, dither_col]
                    i= -1
                    # for _ in range(0,3):
                    #     i+= 1
                    #     if new_val[i] < 0:
                    #         new_val[i] = 0
                    #     elif new_val[i] > 255:
                    #         new_val[i] = 255
                    img[row+dither_row, col+dither_col - anchor,0] = new_r
                    img[row+dither_row, col+dither_col - anchor,1] = new_g
                    img[row+dither_row, col+dither_col - anchor,2] = new_b
                    # img[row+dither_row, col+dither_col - anchor,1] = new_g
                    # img[row+dither_row, col+dither_col - anchor,2] = new_b
                    # img[row+dither_row, col+dither_col - anchor] = new_val.astype(np.uint8)
                    # img[row+dither_row, col+dither_col] = new_val.astype(np.uint8)

    return color_table_indices

def find_closest_color(np.ndarray[np.uint8_t,ndim=1] pixel, np.ndarray[np.uint8_t,ndim=2] color_table):
    cdef int rows = color_table.shape[0]
    cdef int cols = color_table.shape[1]
    cdef int min_distance = 255*255+1
    cdef int min_index = 0
    cdef int distance = 0
    cdef int r_var = 0
    cdef int g_var = 0
    cdef int b_var = 0
    cdef int pixel_r = pixel[0]
    cdef int pixel_g = pixel[1]
    cdef int pixel_b = pixel[2]
    cdef int i = -1

    for _ in range(0,rows):
        i+= 1
        r_var = pixel_r - color_table[i,0]
        g_var = pixel_g - color_table[i,1]
        b_var = pixel_b - color_table[i,2]
        distance = r_var*r_var + g_var*g_var + b_var*b_var
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index