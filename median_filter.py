"""
median_filter.py

Ring buffer where frames can be added sequentially, and median calculated
along the temporal axis - for background subtraction

Contains two implementations:
 - NumPy-based version that uses CPU
 - PyCUDA-based version that uses GPU, if available

Main interfaces:
 - new_frame: stores a new frame in the buffer
 - calculate_bg: calculates the temporal median along the frame buffer
 - abs_diff: calculates the absolute difference between the last frame, 
    and the last calculated background

Draws heavily on work by Robert Crovella (NVIDIA):
https://forums.developer.nvidia.com/t/median-filter-time-dimension-for-images-bad-performance-no-memory-coalescence/36103
"""

import time
import importlib.util
from abc import ABC

import numpy as np

# Check if PyCuda is installed
package_name = 'pycuda'
use_cuda = importlib.util.find_spec(package_name) is not None

if use_cuda:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule


# Abstract class for frame buffer
class MedianBufferABC(ABC):

    def new_frame(self):
        raise NotImplementedError

    def calculate_bg(self):
        raise NotImplementedError

    def abs_diff(self):
        raise NotImplementedError

    def last_frame(self):
        raise NotImplementedError


# CUDA code in this string contains template keywords that must be replaced with actual values, e.g. __IMW_keyword__
median_21_func_def = """

#define IMW __IMW_keyword__   // image width (px)
#define IMH __IMH_keyword__   // image height (px)
#define BX __BX_keyword__     // block index x
#define BY __BY_keyword__     // block index y
#define nTPB (BX*BY)          // threads per block
#define IMN 21                // buffer length

#define G(A,B) (array[(A*nTPB)+(threadIdx.y*BX)+threadIdx.x]>array[(B*nTPB)+(threadIdx.y*BX)+threadIdx.x])
#define E(A,B) (array[(A*nTPB)+(threadIdx.y*BX)+threadIdx.x]>=array[(B*nTPB)+(threadIdx.y*BX)+threadIdx.x])

__global__  void median_filter( unsigned __DTYPE_keyword__* frames, unsigned __DTYPE_keyword__* result)
{
  __shared__ unsigned __DTYPE_keyword__ array[IMN*nTPB];
  int x = threadIdx.x+blockDim.x*blockIdx.x;
  int y = threadIdx.y+blockDim.y*blockIdx.y;
  int num = x + y * (IMW);

  if ((x < IMW) && (y < IMH)){
    for (int i = 0; i < IMN; i++)
      array[(i*nTPB)+(threadIdx.y*BX)+threadIdx.x] = frames[(i * (IMW*IMH)) + ( num)];
    int ot;
    if (10 == G(0,1) +G(0,2) +G(0,3) +G(0,4) +G(0,5) +G(0,6) +G(0,7) +G(0,8) +G(0,9) +G(0,10) +G(0,11) +G(0,12) +G(0,13) +G(0,14) +G(0,15) +G(0,16) +G(0,17) +G(0,18) +G(0,19) +G(0,20))
      ot = 0;
    else if (10 == E(1,0) +G(1,2) +G(1,3) +G(1,4) +G(1,5) +G(1,6) +G(1,7) +G(1,8) +G(1,9) +G(1,10) +G(1,11) +G(1,12) +G(1,13) +G(1,14) +G(1,15) +G(1,16) +G(1,17) +G(1,18) +G(1,19) +G(1,20))
      ot = 1;
    else if (10 == E(2,0) +E(2,1) +G(2,3) +G(2,4) +G(2,5) +G(2,6) +G(2,7) +G(2,8) +G(2,9) +G(2,10) +G(2,11) +G(2,12) +G(2,13) +G(2,14) +G(2,15) +G(2,16) +G(2,17) +G(2,18) +G(2,19) +G(2,20))
      ot = 2;
    else if (10 == E(3,0) +E(3,1) +E(3,2) +G(3,4) +G(3,5) +G(3,6) +G(3,7) +G(3,8) +G(3,9) +G(3,10) +G(3,11) +G(3,12) +G(3,13) +G(3,14) +G(3,15) +G(3,16) +G(3,17) +G(3,18) +G(3,19) +G(3,20))
      ot = 3;
    else if (10 == E(4,0) +E(4,1) +E(4,2) +E(4,3) +G(4,5) +G(4,6) +G(4,7) +G(4,8) +G(4,9) +G(4,10) +G(4,11) +G(4,12) +G(4,13) +G(4,14) +G(4,15) +G(4,16) +G(4,17) +G(4,18) +G(4,19) +G(4,20))
      ot = 4;
    else if (10 == E(5,0) +E(5,1) +E(5,2) +E(5,3) +E(5,4) +G(5,6) +G(5,7) +G(5,8) +G(5,9) +G(5,10) +G(5,11) +G(5,12) +G(5,13) +G(5,14) +G(5,15) +G(5,16) +G(5,17) +G(5,18) +G(5,19) +G(5,20))
      ot = 5;
    else if (10 == E(6,0) +E(6,1) +E(6,2) +E(6,3) +E(6,4) +E(6,5) +G(6,7) +G(6,8) +G(6,9) +G(6,10) +G(6,11) +G(6,12) +G(6,13) +G(6,14) +G(6,15) +G(6,16) +G(6,17) +G(6,18) +G(6,19) +G(6,20))
      ot = 6;
    else if (10 == E(7,0) +E(7,1) +E(7,2) +E(7,3) +E(7,4) +E(7,5) +E(7,6) +G(7,8) +G(7,9) +G(7,10) +G(7,11) +G(7,12) +G(7,13) +G(7,14) +G(7,15) +G(7,16) +G(7,17) +G(7,18) +G(7,19) +G(7,20))
      ot = 7;
    else if (10 == E(8,0) +E(8,1) +E(8,2) +E(8,3) +E(8,4) +E(8,5) +E(8,6) +E(8,7) +G(8,9) +G(8,10) +G(8,11) +G(8,12) +G(8,13) +G(8,14) +G(8,15) +G(8,16) +G(8,17) +G(8,18) +G(8,19) +G(8,20))
      ot = 8;
    else if (10 == E(9,0) +E(9,1) +E(9,2) +E(9,3) +E(9,4) +E(9,5) +E(9,6) +E(9,7) +E(9,8) +G(9,10) +G(9,11) +G(9,12) +G(9,13) +G(9,14) +G(9,15) +G(9,16) +G(9,17) +G(9,18) +G(9,19) +G(9,20))
      ot = 9;
    else if (10 == E(10,0)+E(10,1)+E(10,2)+E(10,3)+E(10,4)+E(10,5)+E(10,6)+E(10,7)+E(10,8)+E(10,9) +G(10,11)+G(10,12)+G(10,13)+G(10,14)+G(10,15)+G(10,16)+G(10,17)+G(10,18)+G(10,19)+G(10,20))
      ot = 10;
    else if (10 == E(11,0)+E(11,1)+E(11,2)+E(11,3)+E(11,4)+E(11,5)+E(11,6)+E(11,7)+E(11,8)+E(11,9) +E(11,10)+G(11,12)+G(11,13)+G(11,14)+G(11,15)+G(11,16)+G(11,17)+G(11,18)+G(11,19)+G(11,20))
      ot = 11;
    else if (10 == E(12,0)+E(12,1)+E(12,2)+E(12,3)+E(12,4)+E(12,5)+E(12,6)+E(12,7)+E(12,8)+E(12,9) +E(12,10)+E(12,11)+G(12,13)+G(12,14)+G(12,15)+G(12,16)+G(12,17)+G(12,18)+G(12,19)+G(12,20))
      ot = 12;
    else if (10 == E(13,0)+E(13,1)+E(13,2)+E(13,3)+E(13,4)+E(13,5)+E(13,6)+E(13,7)+E(13,8)+E(13,9) +E(13,10)+E(13,11)+E(13,12)+G(13,14)+G(13,15)+G(13,16)+G(13,17)+G(13,18)+G(13,19)+G(13,20))
      ot = 13;
    else if (10 == E(14,0)+E(14,1)+E(14,2)+E(14,3)+E(14,4)+E(14,5)+E(14,6)+E(14,7)+E(14,8)+E(14,9) +E(14,10)+E(14,11)+E(14,12)+E(14,13)+G(14,15)+G(14,16)+G(14,17)+G(14,18)+G(14,19)+G(14,20))
      ot = 14;
    else if (10 == E(15,0)+E(15,1)+E(15,2)+E(15,3)+E(15,4)+E(15,5)+E(15,6)+E(15,7)+E(15,8)+E(15,9) +E(15,10)+E(15,11)+E(15,12)+E(15,13)+E(15,14)+G(15,16)+G(15,17)+G(15,18)+G(15,19)+G(15,20))
      ot = 15;
    else if (10 == E(16,0)+E(16,1)+E(16,2)+E(16,3)+E(16,4)+E(16,5)+E(16,6)+E(16,7)+E(16,8)+E(16,9) +E(16,10)+E(16,11)+E(16,12)+E(16,13)+E(16,14)+E(16,15)+G(16,17)+G(16,18)+G(16,19)+G(16,20))
      ot = 16;
    else if (10 == E(17,0)+E(17,1)+E(17,2)+E(17,3)+E(17,4)+E(17,5)+E(17,6)+E(17,7)+E(17,8)+E(17,9) +E(17,10)+E(17,11)+E(17,12)+E(17,13)+E(17,14)+E(17,15)+E(17,16)+G(17,18)+G(17,19)+G(17,20))
      ot = 17;
    else if (10 == E(18,0)+E(18,1)+E(18,2)+E(18,3)+E(18,4)+E(18,5)+E(18,6)+E(18,7)+E(18,8)+E(18,9) +E(18,10)+E(18,11)+E(18,12)+E(18,13)+E(18,14)+E(18,15)+E(18,16)+E(18,17)+G(18,19)+G(18,20))
      ot = 18;
    else if (10 == E(19,0)+E(19,1)+E(19,2)+E(19,3)+E(19,4)+E(19,5)+E(19,6)+E(19,7)+E(19,8)+E(19,9) +E(19,10)+E(19,11)+E(19,12)+E(19,13)+E(19,14)+E(19,15)+E(19,16)+E(19,17)+E(19,18)+G(19,20))
      ot = 19;
    else
      ot = 20;
    result[(num)] = array[(ot*nTPB) + (threadIdx.y*BX)+threadIdx.x];
  }
}

__global__  void absDiff( unsigned __DTYPE_keyword__* frame1, unsigned __DTYPE_keyword__* frame2, unsigned __DTYPE_keyword__* result)
{
    // Calculate the absolute difference between the last calculated median (the background), and the last frame
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    int pixel_num = x + y * (IMW);              // pixel offset

    if ((x < IMW) && (y < IMH)){
        result[pixel_num] = __usad( frame1[pixel_num], frame2[pixel_num], 0 );
    }
}

"""


class _MedianBufferNumPy(MedianBufferABC):
    """
    Frame buffer implemented using NumPy - no CUDA required
    """

    buffer_length = 21  # hard-coded: cannot change

    def __init__(self, image_width=1024, image_height=1, bx=None, by=None, oneD=False, dtype=np.uint16) -> None:

        self.image_width = image_width
        self.image_height = image_height
        self.bx = bx
        self.by = by
        self.frame_size = image_width * image_height
        self.oneD = oneD  # if True, flatten outputs to vector
        self.dtype = dtype

        print(f'image_width: {self.image_width}, image_height: {self.image_height}, bx: {self.bx}, by: {self.by}, frame_size: {self.frame_size}, dtype: {str(self.dtype)}, CUDA: {False}')

        self._idx = -1  # internal frame counter - the most recent frame that HAS been added
        self.last_frame = np.zeros(
            shape=(self.image_width, self.image_height), dtype=self.dtype)
        # create frame buffer
        self.frame_buffer = np.zeros(shape=(
            self.buffer_length, self.image_width, self.image_height), dtype=self.dtype)
        self._background = np.zeros(
            shape=(self.image_width, self.image_height), dtype=self.dtype)

    def new_frame(self, frame: np.ndarray, addToBuffer=True):
        """
        Add a new frame to the buffer, at the offset (if) specified
        """

        # print(f'frame.shape: {frame.shape}')
        # assert frame.shape == (self.image_width, self.image_height)

        frame = frame.astype(self.dtype)

        if addToBuffer:
            # Advance internal frame counter
            self._idx = (self._idx+1) % self.buffer_length
            self.frame_buffer[self._idx] = frame.reshape(
                (self.image_width, self.image_height))  # TODO: avoid this reshape operation?
        else:
            self.last_frame = frame

    def calculate_bg(self):

        self._background = np.median(
            self.frame_buffer, axis=0).astype(self.frame_buffer.dtype)

    def _formatOutput(self, output: np.ndarray):
        """ 
        If working in 1-D, vectorise the output
        """

        if self.oneD:
            return output.flatten()
        else:
            return output

    def getBackground(self):
        return self._formatOutput(self._background)

    def abs_diff(self):
        return self._formatOutput(np.abs(self._background-self.last_frame))

    def last_frame(self):
        return self._formatOutput(self.last_frame)


class _MedianBufferCUDA(MedianBufferABC):
    """
    CUDA implementation of the frame buffer
    """

    # These keywords are replaced by configuration parameters
    image_width_keyword = "__IMW_keyword__"
    image_height_keyword = "__IMH_keyword__"
    block_x_keyword = "__BX_keyword__"
    block_y_keyword = "__BY_keyword__"
    dtype_keyword = "__DTYPE_keyword__"

    dtype_map = {np.uint8: 'char', np.uint16: 'short'}

    buffer_length = 21  # hard-coded: cannot change as the sorting network is pre-defined

    def __init__(self, image_width=1024, image_height=1, bx=32, by=1, oneD=False, dtype=np.uint16) -> None:

        # force to int to avoid parameters as numpy.int64
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.bx = int(bx)
        self.by = int(by)
        self.oneD = oneD  # if True, flatten outputs to vector
        self.dtype = dtype

        # Create CUDA module and set parameters
        mod = SourceModule(
            median_21_func_def
            .replace(self.image_width_keyword,  str(self.image_width))
            .replace(self.image_height_keyword, str(self.image_height))
            .replace(self.block_x_keyword,      str(self.bx))
            .replace(self.block_y_keyword,      str(self.by))
            .replace(self.dtype_keyword,        str(self.dtype_map[self.dtype]))
        )
        self.median_func = mod.get_function("median_filter")
        self.absDiff_func = mod.get_function("absDiff")

        self._idx = -1  # internal frame counter - the most recent frame that HAS been added
        self.last_frame = np.zeros(
            shape=(self.image_width, self.image_height), dtype=self.dtype)

        # create frame buffer
        self.frame_buffer = np.zeros(shape=(
            self.buffer_length, self.image_width, self.image_height), dtype=self.dtype)
        self._background = np.zeros(
            shape=(self.image_width, self.image_height), dtype=self.dtype)
        self._absDiff = np.zeros(
            shape=(self.image_width, self.image_height), dtype=self.dtype)

        # create gpu variables
        self.last_frame_gpu = cuda.mem_alloc(self.last_frame.nbytes)
        self.frame_buffer_gpu = cuda.mem_alloc(self.frame_buffer.nbytes)
        self.background_gpu = cuda.mem_alloc(self._background.nbytes)
        self.absDiff_gpu = cuda.mem_alloc(self._absDiff.nbytes)

        self.frame_size = image_width * image_height
        self.frame_size_bytes = self._background.nbytes

        print(f'image_width: {self.image_width}, image_height: {self.image_height}, bx: {self.bx}, by: {self.by}, frame_size: {self.frame_size}, dtype: {str(self.dtype)}, CUDA: {True}')

    def new_frame(self, frame: np.ndarray, addToBuffer=True):
        """
        Add a new frame and transfer to the GPU
         - if addToBuffer is True, frame is uploaded to the frame buffer for calculating the median, 
         - otherwise, frame is uploaded into last_frame, ready for comparing to the background image.
        """

        frame = frame.astype(self.dtype)

        # copy new frame to GPU
        if addToBuffer:
            # Advance internal frame counter
            self._idx = (self._idx+1) % self.buffer_length
            buffer_offset = int(
                (self._idx % self.buffer_length) * self.frame_size_bytes)
            cuda.memcpy_htod(int(self.frame_buffer_gpu) + buffer_offset, frame)
        else:
            self.last_frame = frame
            cuda.memcpy_htod(self.last_frame_gpu, frame)

    def calculate_bg(self):
        """
        Calculate median along the buffer
        """
        # run median filter on GPU
        self.median_func(
            self.frame_buffer_gpu,
            self.background_gpu,
            block=(self.bx, self.by, 1),
            grid=(self.image_width//self.bx, self.image_height//self.by, 1)
        )

    def _formatOutput(self, output: np.ndarray):
        """ 
        If working in 1-D, vectorise the output
        """

        if self.oneD:
            return output.flatten()
        else:
            return output

    def getBackground(self):
        """
        Returns the background image

        Avoid using this function if possible: leave the data on the GPU and do stuff there as much as possbile
        """

        # Get data back from GPU
        cuda.memcpy_dtoh(self._background, self.background_gpu)

        return self._formatOutput(self._background)

    def abs_diff(self):
        """
        Calculates the absolute difference between last_frame and the background (median of buffer)
        """
        # run absDiff filter on GPU
        self.absDiff_func(
            self.background_gpu,    # background
            self.last_frame_gpu,    # last frame received
            self.absDiff_gpu,       # result
            block=(self.bx, self.by, 1),
            grid=(self.image_width//self.bx, self.image_height//self.by, 1)
        )

        # Get data back
        cuda.memcpy_dtoh(self._absDiff, self.absDiff_gpu)

        return self._formatOutput(self._absDiff)

    def last_frame(self):
        return self._formatOutput(self.last_frame)


# Determine which implementation to use, depending on availability of PyCuda
if use_cuda:
    MedianBuffer = _MedianBufferCUDA
else:
    MedianBuffer = _MedianBufferNumPy


if __name__ == '__main__':
    # Run some performance benchmarks, comparing the two implementations

    print(f'Median_filter - using CUDA: {use_cuda}')

    im_x, im_y = 128, 128
    # im_x, im_y = 1024, 1024
    dtype = np.uint16

    # Create the frame buffer
    buff1 = MedianBuffer(im_x, im_y, bx=32, by=16, dtype=dtype)

    t1 = time.time()
    report_every = 500
    repeats = 5
    total_cycles = repeats*report_every

    # Create all the image frames in advance using random numbers
    all_frames = np.random.randint(
        0, 255*255, size=(total_cycles, buff1.image_width, buff1.image_height)).astype(buff1.dtype)

    for i in range(1, total_cycles+1):

        # Add frame to buffer
        this_frame = all_frames[i-1]
        buff1.new_frame(this_frame, addToBuffer=i % 2)

        # Update the background using the median calculation
        buff1.calculate_bg()

        # Calculate the absolute difference between the current
        # frame and the new background, and get the result back
        absDiff1 = buff1.abs_diff()

        if (i % report_every) == 0:
            print(f'{report_every / (time.time()-t1):0.2f} cycles/sec')
            t1 = time.time()

    bg1 = buff1.getBackground()

    if use_cuda:
        # Repeat, without CUDA
        print(f'Median_filter - without CUDA')

        buff2 = _MedianBufferNumPy(im_x, im_y, bx=32, by=16, dtype=dtype)
        t1 = time.time()

        for i in range(1, total_cycles+1):

            # Add frame to buffer
            this_frame = all_frames[i-1]
            buff2.new_frame(this_frame, addToBuffer=i % 2)

            # Update the background using the median calculation
            buff2.calculate_bg()

            # Calculate the absolute difference between the current
            # frame and the new background, and get the result back
            absDiff2 = buff2.abs_diff()

            if (i % report_every) == 0:
                print(f'{report_every / (time.time()-t1):0.2f} cycles/sec')
                t1 = time.time()

        # Compare background results to make sure they are identical
        bg2 = buff2.getBackground()
        assert np.array_equal(bg1.shape, bg2.shape)
        assert np.array_equal(bg1, bg2)
        print('Backgrounds (median) match!')

        # Compare absDiff results
        # print(absDiff1.shape, absDiff2.shape)
        # print(absDiff1, absDiff2)
        # I = absDiff1.flatten()
        # J = absDiff2.flatten()
        # for idx in range(absDiff1.size):
        #     if not I[idx]==J[idx]:
        #         print(idx, I[idx], J[idx])

        assert np.array_equal(absDiff1, absDiff2)
        print('Abs diff results match!')
