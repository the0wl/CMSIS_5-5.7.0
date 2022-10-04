/* ----------------------------------------------------------------------
* Copyright (C) 2010-2018 Arm Limited. All rights reserved.
*
*
* Project:       CMSIS NN Library
* Title:         arm_nnexamples_cifar10.cpp
*
* Description:   Convolutional Neural Network Example
*
* Target Processor: Cortex-M4/Cortex-M7
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the
*     distribution.
*   - Neither the name of Arm LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
* -------------------------------------------------------------------- */

/**
 * @ingroup groupExamples
 */

/**
 * @defgroup CNNExample Convolutional Neural Network Example
 *
 * \par Description:
 * \par
 * Demonstrates a convolutional neural network (CNN) example with the use of convolution,
 * ReLU activation, pooling and fully-connected functions.
 *
 * \par Model definition:
 * \par
 * The CNN used in this example is based on CIFAR-10 example from Caffe [1]. 
 * The neural network consists
 * of 3 convolution layers interspersed by ReLU activation and max pooling layers, followed by a 
 * fully-connected layer at the end. The input to the network is a 32x32 pixel color image, which will 
 * be classified into one of the 10 output classes. 
 * This example model implementation needs 32.3 KB to store weights, 40 KB for activations and 
 * 3.1 KB for storing the \c im2col data.
 *
 * \image html CIFAR10_CNN.gif "Neural Network model definition"
 *
 * \par Variables Description:
 * \par
 * \li \c conv1_wt, \c conv2_wt, \c conv3_wt are convolution layer weight matrices
 * \li \c conv1_bias, \c conv2_bias, \c conv3_bias are convolution layer bias arrays
 * \li \c ip1_wt, ip1_bias point to fully-connected layer weights and biases
 * \li \c input_data points to the input image data
 * \li \c output_data points to the classification output
 * \li \c col_buffer is a buffer to store the \c im2col output
 * \li \c scratch_buffer is used to store the activation data (intermediate layer outputs)
 *
 * \par CMSIS DSP Software Library Functions Used:
 * \par
 * - arm_convolve_HWC_q7_RGB()
 * - arm_convolve_HWC_q7_fast()
 * - arm_relu_q7()
 * - arm_maxpool_q7_HWC()
 * - arm_avepool_q7_HWC()
 * - arm_fully_connected_q7_opt()
 * - arm_fully_connected_q7()
 *
 * <b> Refer  </b>
 * \link arm_nnexamples_cifar10.cpp \endlink
 *
 * \par [1] https://github.com/BVLC/caffe
 */

#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include "../../../../../DSP/Include/arm_math.h"
#include "arm_nnexamples_cifar10_parameter.h"
#include "arm_nnexamples_cifar10_weights.h"

#include "../../../../Include/arm_nnfunctions.h"
//#include "arm_nnexamples_cifar10_inputs.h"
#include "custom_cifar_10_inputs.h"

#ifdef _RTE_
#include "RTE_Components.h"
#ifdef RTE_Compiler_EventRecorder
#include "EventRecorder.h"
#endif
#endif

// include the input and weights

static q7_t conv1_wt[CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

static q7_t conv2_wt[CONV2_IM_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IM_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

static q7_t ip1_wt[IP1_DIM * IP1_OUT] = IP1_WT;
static q7_t ip1_bias[IP1_OUT] = IP1_BIAS;

/* Here the image_data should be the raw uint8 type RGB image in [RGB, RGB, RGB ... RGB] format */
uint8_t   image_data[1000][CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA;
q7_t      output_data[IP1_OUT];

//vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
q7_t      col_buffer[2 * 5 * 5 * 32 * 2];

q7_t      scratch_buffer[32 * 32 * 10 * 4];

void retornaClasse(const int k_image, const q7_t * resultado);

int main()
{
  #ifdef RTE_Compiler_EventRecorder
  EventRecorderInitialize (EventRecordAll, 1);  // initialize and start Event Recorder
  #endif

  printf("start execution\n");
  /* start the execution */

  q7_t     *img_buffer1 = scratch_buffer;
  q7_t     *img_buffer2 = img_buffer1 + 32 * 32 * 32;

  int numThreads = 1;
  printf("Informe o numero de Threads (1-4): ");
  scanf("%d", &numThreads);

  if (numThreads < 1)
    numThreads =  1;
    
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);

  #pragma omp parallel for num_threads(numThreads)  
//    #pragma omp for  
    for (int j=0; j < 1000; j++) { // KELVIN
    
      /* input pre-processing */
      int mean_data[3] = INPUT_MEAN_SHIFT;
      unsigned int scale_data[3] = INPUT_RIGHT_SHIFT;
      for (int i=0;i<32*32*3; i+=3) {
          img_buffer2[i] =   (q7_t)__SSAT( ((((int)image_data[j][i]   - mean_data[0])<<7) + (0x1<<(scale_data[0]-1)))
                                  >> scale_data[0], 8);
          img_buffer2[i+1] = (q7_t)__SSAT( ((((int)image_data[j][i+1] - mean_data[1])<<7) + (0x1<<(scale_data[1]-1)))
                                  >> scale_data[1], 8);
          img_buffer2[i+2] = (q7_t)__SSAT( ((((int)image_data[j][i+2] - mean_data[2])<<7) + (0x1<<(scale_data[2]-1)))
                                  >> scale_data[2], 8);
      }
      
      // conv1 img_buffer2 -> img_buffer1
      arm_convolve_HWC_q7_RGB(img_buffer2, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
                              CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, img_buffer1, CONV1_OUT_DIM,
                              (q15_t *) col_buffer, NULL);

      arm_relu_q7(img_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);

      // pool1 img_buffer1 -> img_buffer2
      arm_maxpool_q7_HWC(img_buffer1, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
                          POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, img_buffer2);

      // conv2 img_buffer2 -> img_buffer1
      arm_convolve_HWC_q7_fast(img_buffer2, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM,
                              CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, img_buffer1,
                              CONV2_OUT_DIM, (q15_t *) col_buffer, NULL);

      arm_relu_q7(img_buffer1, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);

      // pool2 img_buffer1 -> img_buffer2
      arm_maxpool_q7_HWC(img_buffer1, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
                          POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, img_buffer2);

      // conv3 img_buffer2 -> img_buffer1
      arm_convolve_HWC_q7_fast(img_buffer2, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM,
                              CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, img_buffer1,
                              CONV3_OUT_DIM, (q15_t *) col_buffer, NULL);

      arm_relu_q7(img_buffer1, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);

      // pool3 img_buffer-> img_buffer2
      arm_maxpool_q7_HWC(img_buffer1, CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM,
                          POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, img_buffer2);


      arm_fully_connected_q7_opt(img_buffer2, ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias,
                                  output_data, (q15_t *) img_buffer1);
      
      arm_softmax_q7(output_data, 10, output_data);
      
      // #pragma omp critical
      // {
        retornaClasse(j, output_data);
      // }
    }
    //  }
  

    gettimeofday(&tv2, NULL);

    printf ("Total time = %f seconds\n",
            (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
            (double) (tv2.tv_sec - tv1.tv_sec));
  return 0;
}

void retornaClasse(const int k_image, const q7_t * resultado) {
  int wMaior = 0;
  int wClasse = -1;

  for (int i = 0; i < 10; i++)
  {
    if (resultado[i] > wMaior) {
      wMaior = resultado[i];
      wClasse = i;
    }
  }

  switch(wClasse) {
  case 0:
    printf("A imagem %03d é da classe aviao [ %d ]\n", k_image, wMaior);
    break;
  case 1:
    printf("A imagem %03d é da classe automovel [ %d ]\n", k_image, wMaior);
    break;
  case 2:
    printf("A imagem %03d é da classe passaro [ %d ]\n", k_image, wMaior);
    break;
  case 3:
    printf("A imagem %03d é da classe gato [ %d ]\n", k_image, wMaior);
    break;
  case 4:
    printf("A imagem %03d é da classe viado [ %d ]\n", k_image, wMaior);
    break;
  case 5:
    printf("A imagem %03d é da classe cachorro [ %d ]\n", k_image, wMaior);
    break;
  case 6:
    printf("A imagem %03d é da classe sápo [ %d ]\n", k_image, wMaior);
    break;
  case 7:
    printf("A imagem %03d é da classe cavalo [ %d ]\n", k_image, wMaior);
    break;
  case 8:
    printf("A imagem %03d é da classe navio [ %d ]\n", k_image, wMaior);
    break;
  case 9:
    printf("A imagem %03d é da classe caminhao [ %d ]\n", k_image, wMaior);
    break;
  default:
    printf("A imagem %03d nao pode ser classificada\n", k_image);
  }
} 

arm_status
arm_convolve_HWC_q7_RGB(const q7_t * Im_in,
                        const uint16_t dim_im_in,
                        const uint16_t ch_im_in,
                        const q7_t * wt,
                        const uint16_t ch_im_out,
                        const uint16_t dim_kernel,
                        const uint16_t padding,
                        const uint16_t stride,
                        const q7_t * bias,
                        const uint16_t bias_shift,
                        const uint16_t out_shift,
                        q7_t * Im_out, const uint16_t dim_im_out, q15_t * bufferA, q7_t * bufferB)
{
    (void)bufferB;
#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */
    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;

    /*
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */
    q15_t    *pBuffer = bufferA;
    q7_t     *pOut = Im_out;

    // check if number of input channels is 3
    if (ch_im_in != 3)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }
    // This part implements the im2col function
    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* Equivalent to arm_fill_q15(0, pBuffer, ch_im_in) with assumption: ch_im_in = 3 */
                        *__SIMD32(pBuffer) = 0x0;
                        *(pBuffer + 2) = 0;
                        pBuffer += 3;
                    } else
                    {
                        /*
                         * Equivalent to:
                         *  arm_q7_to_q15_no_shift( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*3, pBuffer, 3);
                         */

                        const q7_t *pPixel = Im_in + (i_ker_y * dim_im_in + i_ker_x) * 3;
                        q31_t     buf = arm_nn_read_q7x4(pPixel);

                        union arm_nnword top;
                        union arm_nnword bottom;

                        top.word = __SXTB16(buf);
                        bottom.word = __SXTB16(__ROR(buf, 8));

#ifndef ARM_MATH_BIG_ENDIAN
                        /*
                         *  little-endian, | omit | 3rd  | 2nd  | 1st  |
                         *                MSB                         LSB
                         *   top | 3rd | 1st |; bottom | omit | 2nd |
                         *
                         *  version 1, need to swap 2nd and 3rd weight
                         * *__SIMD32(pBuffer) = top.word;
                         * *(pBuffer+2) = bottom.half_words[0];
                         *
                         *  version 2, no weight shuffling required
                         */
                        *pBuffer++ = top.half_words[0];
                        *__SIMD32(pBuffer) = __PKHBT(bottom.word, top.word, 0);
#else
                        /*
                         *  big-endian,    | 1st  | 2nd  | 3rd  | omit |
                         *                MSB                         LSB
                         *  top | 2nd | omit |; bottom | 1st | 3rd |
                         *
                         *  version 1, need to swap 2nd and 3rd weight
                         * *__SIMD32(pBuffer) = bottom.word;
                         * *(pBuffer+2) = top.half_words[1];
                         *
                         *  version 2, no weight shuffling required
                         */
                        *pBuffer++ = bottom.half_words[0];
                        *__SIMD32(pBuffer) = __PKHTB(top.word, bottom.word, 0);
#endif
                        pBuffer += 2;
                    }
                }
            }

            if (pBuffer == bufferA + 2 * 3 * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15(wt, bufferA,
                                                  ch_im_out,
                                                  3 * dim_kernel * dim_kernel, bias_shift, out_shift, bias, pOut);

                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* left-over because odd number of output pixels */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;
        int       i;

        for (i = 0; i < ch_im_out; i++)
        {
            q31_t     sum = ((q31_t)bias[i] << bias_shift) + NN_ROUND(out_shift);
            q15_t    *pB = bufferA;
            /* basically each time it process 4 entries */
            uint16_t  colCnt = 3 * dim_kernel * dim_kernel >> 2;

            while (colCnt)
            {

                q31_t     inA1, inA2;
                q31_t     inB1, inB2;

                pA = read_and_pad(pA, &inA1, &inA2);

                inB1 = arm_nn_read_q15x2_ia((const q15_t **)&pB);
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = arm_nn_read_q15x2_ia((const q15_t **)&pB);
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = 3 * dim_kernel * dim_kernel & 0x3;
            while (colCnt)
            {
                q7_t      inA1 = *pA++;
                q15_t     inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            *pOut++ = (q7_t) __SSAT((sum >> out_shift), 8);
        }
    }
#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    signed char in_row, in_col;

    // check if number of input channels is 3
    if (ch_im_in != 3)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = (bias[i] << bias_shift) + NN_ROUND(out_shift);
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        /* if-for implementation */
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    Im_in[(in_row * dim_im_in + in_col) * ch_im_in +
                                          l] * wt[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel +
                                                                                            n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out + k) * ch_im_out] = (q7_t) __SSAT((conv_out >> out_shift), 8);
            }
        }
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return (ARM_MATH_SUCCESS);
}

void arm_relu_q7(q7_t *data, uint16_t size)
{

#if defined(ARM_MATH_DSP)
    /* Run the following code for M cores with DSP extension */

    uint16_t i = size >> 2;
    q7_t *input = data;
    q7_t *output = data;
    q31_t in;
    q31_t buf;
    q31_t mask;

    while (i)
    {
        in = read_q7x4_ia(&input);

        /* extract the first bit */
        buf = __ROR(in & 0x80808080, 7);

        /* if MSB=1, mask will be 0xFF, 0x0 otherwise */
        mask = __QSUB8(0x00000000, buf);

        write_q7x4_ia(&output, in & (~mask));

        i--;
    }

    i = size & 0x3;
    while (i)
    {
        if (*input < 0)
        {
            *input = 0;
        }
        input++;
        i--;
    }

#else
    /* Run the following code as reference implementation for cores without DSP extension */

    uint16_t i;

    for (i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }

#endif
}

void
arm_maxpool_q7_HWC(q7_t * Im_in,
                   const uint16_t dim_im_in,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel,
                   const uint16_t padding,
                   const uint16_t stride, const uint16_t dim_im_out, q7_t * bufferA, q7_t * Im_out)
{
    (void)bufferA;
#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_x, i_y;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < dim_im_in; i_y++)
    {

        for (i_x = 0; i_x < dim_im_out; i_x++)
        {
            /* for each output pixel */
            q7_t     *target = Im_in + (i_y * dim_im_in + i_x) * ch_im_in;
            q7_t     *win_start;
            q7_t     *win_stop;
            if (i_x * stride - padding < 0)
            {
                win_start = target;
            } else
            {
                win_start = Im_in + (i_y * dim_im_in + i_x * stride - padding) * ch_im_in;
            }

            if (i_x * stride - padding + dim_kernel >= dim_im_in)
            {
                win_stop = Im_in + (i_y * dim_im_in + dim_im_in) * ch_im_in;
            } else
            {
                win_stop = Im_in + (i_y * dim_im_in + i_x * stride - padding + dim_kernel) * ch_im_in;
            }

            /* first step is to copy over initial data */
            /* arm_copy_q7(win_start, target, ch_im_in); */
            memmove(target, win_start, ch_im_in);

            /* start the max operation from the second part */
            win_start += ch_im_in;
            for (; win_start < win_stop; win_start += ch_im_in)
            {
                compare_and_replace_if_larger_q7(target, win_start, ch_im_in);
            }
        }
    }

    /* then does the pooling along y axis */
    for (i_y = 0; i_y < dim_im_out; i_y++)
    {

        /* for each output row */
        q7_t     *target = Im_out + i_y * dim_im_out * ch_im_in;
        q7_t     *row_start;
        q7_t     *row_end;
        /* setting the starting row */
        if (i_y * stride - padding < 0)
        {
            row_start = Im_in;
        } else
        {
            row_start = Im_in + (i_y * stride - padding) * dim_im_in * ch_im_in;
        }
        /* setting the stopping row */
        if (i_y * stride - padding + dim_kernel >= dim_im_in)
        {
            row_end = Im_in + dim_im_in * dim_im_in * ch_im_in;
        } else
        {
            row_end = Im_in + (i_y * stride - padding + dim_kernel) * dim_im_in * ch_im_in;
        }

        /* copy over the first row */
        /* arm_copy_q7(row_start, target, dim_im_out * ch_im_in); */
        memmove(target, row_start, dim_im_out * ch_im_in);

        /* move over to next row */
        row_start += ch_im_in * dim_im_in;

        for (; row_start < row_end; row_start += dim_im_in * ch_im_in)
        {
            compare_and_replace_if_larger_q7(target, row_start, dim_im_out * ch_im_in);
        }
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int       max = -129;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = max;
            }
        }
    }

#endif                          /* ARM_MATH_DSP */

}

arm_status
arm_convolve_HWC_q7_fast(const q7_t * Im_in,
                         const uint16_t dim_im_in,
                         const uint16_t ch_im_in,
                         const q7_t * wt,
                         const uint16_t ch_im_out,
                         const uint16_t dim_kernel,
                         const uint16_t padding,
                         const uint16_t stride,
                         const q7_t * bias,
                         const uint16_t bias_shift,
                         const uint16_t out_shift,
                         q7_t * Im_out,
                         const uint16_t dim_im_out,
                         q15_t * bufferA,
                         q7_t * bufferB)
{
    (void)bufferB;
#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;

    /*
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t    *pBuffer = bufferA;
    q7_t     *pOut = Im_out;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Top: i_out_y from 0 to padding - 1
     * Middle: i_out_y from padding to dim_im_out-padding-1
     * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
     */

    /* top part */
    for (i_out_y = 0; i_out_y < padding; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                        arm_q7_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* middle part, here we also divide the x into left, mid and right */
    for (; i_out_y < dim_im_out - padding; i_out_y++)
    {

        /* left part */
        for (i_out_x = 0; i_out_x < padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                        arm_q7_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* mid part */
        for (; i_out_x < dim_im_out - padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                arm_q7_to_q15_reordered_no_shift((q7_t *) Im_in
                                                 +
                                                 (i_ker_y *
                                                  dim_im_in +
                                                  i_out_x *
                                                  stride - padding) * ch_im_in, pBuffer, ch_im_in * dim_kernel);
                pBuffer += ch_im_in * dim_kernel;
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* right part */
        for (; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                        arm_q7_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    for (; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                        arm_q7_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;
        int       i;

        for (i = 0; i < ch_im_out; i++)
        {
            q31_t     sum = ((q31_t)bias[i] << bias_shift) + NN_ROUND(out_shift);
            const q15_t *pB = bufferA;
            /* each time it process 4 entries */
            uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 2;

            while (colCnt)
            {

                q31_t     inA1, inA2;
                q31_t     inB1, inB2;

                pA = read_and_pad_reordered(pA, &inA1, &inA2);

                inB1 = arm_nn_read_q15x2_ia(&pB);
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = arm_nn_read_q15x2_ia(&pB);
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = ch_im_in * dim_kernel * dim_kernel & 0x3;
            while (colCnt)
            {
                q7_t      inA1 = *pA++;
                q15_t     inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            *pOut = (q7_t) __SSAT((sum >> out_shift), 8);
            pOut++;

        }

    }
#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    signed char in_row, in_col;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = (bias[i] << bias_shift) + NN_ROUND(out_shift);
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    Im_in[(in_row * dim_im_in + in_col) * ch_im_in +
                                          l] * wt[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel +
                                                                                            n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out + k) * ch_im_out] = (q7_t) __SSAT((conv_out >> out_shift), 8);
            }
        }
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

arm_status
arm_fully_connected_q7_opt(const q7_t * pV,
                           const q7_t * pM,
                           const uint16_t dim_vec,
                           const uint16_t num_of_rows,
                           const uint16_t bias_shift,
                           const uint16_t out_shift,
                           const q7_t * bias,
                           q7_t * pOut,
                           q15_t * vec_buffer)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    const q7_t *pB = pM;
    q7_t     *pO = pOut;
    const q7_t *pBias = bias;
    const q15_t *pA;
    uint16_t  rowCnt = num_of_rows >> 2;

    arm_q7_to_q15_reordered_no_shift(pV, vec_buffer, dim_vec);

    while (rowCnt)
    {

        q31_t     sum =  ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;

#ifdef USE_INTRINSIC

#ifndef ARM_MATH_BIG_ENDIAN
        while (colCnt)
        {
            q31_t     inM11, inM12, inM13, inM14;
            q31_t     inV;

            inV = arm_nn_read_q15x2_ia(&pA);
            inM11 = arm_nn_read_q7x4_ia(&pB);
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM11, inV, sum);
            sum2 = __SMLAD(inM12, inV, sum2);
            inM13 = arm_nn_read_q7x4_ia(&pB);
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM13, inV, sum3);
            sum4 = __SMLAD(inM14, inV, sum4);

            inV = arm_nn_read_q15x2_ia(&pA);
            inM11 = arm_nn_read_q7x4_ia(&pB);
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM11, inV, sum);
            sum2 = __SMLAD(inM12, inV, sum2);
            inM13 = arm_nn_read_q7x4_ia(&pB);
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM13, inV, sum3);
            sum4 = __SMLAD(inM14, inV, sum4);
            colCnt--;
        }
#else
        while (colCnt)
        {
            q31_t     inM11, inM12, inM13, inM14;
            q31_t     inV;

            inV = arm_nn_read_q15x2_ia(&pA);
            inM11 = arm_nn_read_q7x4_ia(&pB);
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM12, inV, sum);
            sum2 = __SMLAD(inM11, inV, sum2);
            inM13 = arm_nn_read_q7x4_ia(&pB);
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM14, inV, sum3);
            sum4 = __SMLAD(inM13, inV, sum4);

            inV = arm_nn_read_q15x2_ia(&pA);
            inM11 = arm_nn_read_q7x4_ia(&pB);
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM12, inV, sum);
            sum2 = __SMLAD(inM11, inV, sum2);
            inM13 = arm_nn_read_q7x4_ia(&pB);
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM14, inV, sum3);
            sum4 = __SMLAD(inM13, inV, sum4);
            colCnt--;
        }
#endif                          /* ARM_MATH_BIG_ENDIAN */

#else

        /*
         * register needed:
         * loop counter: colCnt
         * accumulators: sum, sum2, sum3, sum4
         * pointers: pB, pA
         * weight data: inM11, inM12, inM13, inM14
         * activation data: inV
         */

#ifndef ARM_MATH_BIG_ENDIAN
        asm volatile ("COL_LOOP_%=:\n"
                      "ldr.w r4, [%[pA]], #8\n"
                      "ldr.w r1, [%[pB]], #16\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r1, %[sum]\n"
                      "smlad %[sum2], r4, r0, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-12]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r3, %[sum3]\n"
                      "smlad %[sum4], r4, r2, %[sum4]\n"
                      "ldr.w r4, [%[pA], #-4]\n"
                      "ldr.w r1, [%[pB], #-8]\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r1, %[sum]\n"
                      "smlad %[sum2], r4, r0, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-4]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r3, %[sum3]\n"
                      "smlad %[sum4], r4, r2, %[sum4]\n"
                      "subs %[colCnt], #1\n"
                      "bne COL_LOOP_%=\n":[sum] "+r"(sum),
                      [sum2] "+r"(sum2),[sum3] "+r"(sum3),
                      [sum4] "+r"(sum4),[pB] "+r"(pB),[pA] "+r"(pA):[colCnt] "r"(colCnt):"r0", "r1", "r2", "r3", "r4");
#else
        asm volatile ("COL_LOOP_%=:\n"
                      "ldr.w r4, [%[pA]], #8\n"
                      "ldr.w r1, [%[pB]], #16\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r0, %[sum]\n"
                      "smlad %[sum2], r4, r1, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-12]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r2, %[sum3]\n"
                      "smlad %[sum4], r4, r3, %[sum4]\n"
                      "ldr.w r4, [%[pA], #-4]\n"
                      "ldr.w r1, [%[pB], #-8]\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r0, %[sum]\n"
                      "smlad %[sum2], r4, r1, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-4]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r2, %[sum3]\n"
                      "smlad %[sum4], r4, r3, %[sum4]\n"
                      "subs %[colCnt], #1\n"
                      "bne COL_LOOP_%=\n":[sum] "+r"(sum),
                      [sum2] "+r"(sum2),[sum3] "+r"(sum3),
                      [sum4] "+r"(sum4),[pB] "+r"(pB),[pA] "+r"(pA):[colCnt] "r"(colCnt):"r0", "r1", "r2", "r3", "r4");
#endif                          /* ARM_MATH_BIG_ENDIAN */

#endif                          /* USE_INTRINSIC */

        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q15_t     inV = *pA++;
            q7_t      inM = *pB++;
            q7_t      inM2 = *pB++;
            q7_t      inM3 = *pB++;
            q7_t      inM4 = *pB++;

            sum += inV * inM;
            sum2 += inV * inM2;
            sum3 += inV * inM3;
            sum4 += inV * inM4;
            colCnt--;
        }                       /* while over colCnt */
        *pO++ = (q7_t) (__SSAT((sum >> out_shift), 8));
        *pO++ = (q7_t) (__SSAT((sum2 >> out_shift), 8));
        *pO++ = (q7_t) (__SSAT((sum3 >> out_shift), 8));
        *pO++ = (q7_t) (__SSAT((sum4 >> out_shift), 8));

        /* adjust the pointers and counters */
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        q31_t     sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;

        while (colCnt)
        {
            q31_t     inV1, inV2, inM11, inM12;

            pB = read_and_pad_reordered(pB, &inM11, &inM12);

            inV1 = arm_nn_read_q15x2_ia(&pA);
            sum = __SMLAD(inV1, inM11, sum);

            inV2 = arm_nn_read_q15x2_ia(&pA);
            sum = __SMLAD(inV2, inM12, sum);

            colCnt--;
        }

        /* left-over of the vector */
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q15_t     inV = *pA++;
            q7_t      inM = *pB++;
            sum += inV * inM;
            colCnt--;
        }

        *pO++ = (q7_t) (__SSAT((sum >> out_shift), 8));

        rowCnt--;
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    uint16_t  rowCnt = num_of_rows >> 2;
    const q7_t *pB = pM;
    const q7_t *pA;
    q7_t     *pO = pOut;
    const q7_t *pBias = bias;

    while (rowCnt)
    {
        q31_t     sum =  ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t  colCnt = dim_vec >> 2;

        pA = pV;

        while (colCnt)
        {
            q7_t      inA1 = *pA++;
            q7_t      inA3 = *pA++;
            q7_t      inA2 = *pA++;
            q7_t      inA4 = *pA++;

            q7_t      inB1 = *pB++;
            q7_t      inB3 = *pB++;
            q7_t      inB2 = *pB++;
            q7_t      inB4 = *pB++;

            sum += inA1 * inB1 + inA2 * inB2;
            sum2 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA1 * inB1 + inA2 * inB2;
            sum4 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum += inA3 * inB1 + inA4 * inB2;
            sum2 += inA3 * inB3 + inA4 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA3 * inB1 + inA4 * inB2;
            sum4 += inA3 * inB3 + inA4 * inB4;

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q7_t      inA = *pA++;
            q7_t      inB = *pB++;
            sum += inA * inB;
            inB = *pB++;
            sum2 += inA * inB;
            inB = *pB++;
            sum3 += inA * inB;
            inB = *pB++;
            sum4 += inA * inB;

            colCnt--;
        }
        *pO++ = (q7_t) __SSAT((sum >> out_shift), 8);
        *pO++ = (q7_t) __SSAT((sum2 >> out_shift), 8);
        *pO++ = (q7_t) __SSAT((sum3 >> out_shift), 8);
        *pO++ = (q7_t) __SSAT((sum4 >> out_shift), 8);

        rowCnt--;
    }

    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        int       ip_out = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        int       j;

        pA = pV;
        for (j = 0; j < dim_vec; j++)
        {
            q7_t      inA = *pA++;
            q7_t      inB = *pB++;
            ip_out += inA * inB;
        }
        *pO++ = (q7_t) __SSAT((ip_out >> out_shift), 8);

        rowCnt--;
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to ARM_MATH_SUCCESS */
    return (ARM_MATH_SUCCESS);

}

#define Q7BITS 8
#define LOG2Q7BITS 3

void arm_softmax_q7(const q7_t * vec_in, const uint16_t dim_vec, q7_t * p_out )
{
#if defined (ARM_MATH_DSP)
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q15_t     base;
    uint16_t blkCnt;

    q31_t in,in1,in2;
    q31_t out1, out2;

    q31_t baseV;
    q31_t shiftV;
    const q31_t pad=0x0d0d0d0d;
    const q7_t *pIn=vec_in;

    base = -128;


    /* We first search for the maximum */

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }


    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - Q7BITS;
    baseV = ((base & 0x0FF) << 24) | ((base & 0x0FF) << 16) | ((base & 0x0FF) << 8) | ((base & 0x0FF));

    sum = 0;

    blkCnt = dim_vec >> 2;

    while(blkCnt)
    {
       in=arm_nn_read_q7x4_ia(&pIn);
       in=__SSUB8(in,baseV);

        in1 = __SXTB16(__ROR(in, 8));

        /* extend remaining two q7_t values to q15_t values */
        in2 = __SXTB16(in);

#ifndef ARM_MATH_BIG_ENDIAN
        out2 = __PKHTB(in1, in2, 16);
        out1 = __PKHBT(in2, in1, 16);
#else
        out1 = __PKHTB(in1, in2, 16);
        out2 = __PKHBT(in2, in1, 16);
#endif


       shiftV = __USAT16(out1,LOG2Q7BITS);
       sum += 0x1 << (shiftV & 0x0FF);
       sum += 0x1 << ((shiftV >> 16) & 0x0FF);

       shiftV = __USAT16(out2,LOG2Q7BITS);
       sum += 0x1 << (shiftV & 0x0FF);
       sum += 0x1 << ((shiftV >> 16) & 0x0FF);

       blkCnt--;
    }

    blkCnt = dim_vec & 3;

    while(blkCnt)
    {
       shift = (uint8_t)__USAT(*pIn++ - base, LOG2Q7BITS);
       sum += 0x1 << shift;
       blkCnt--;
    }


    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;


    pIn=vec_in;

    blkCnt = dim_vec >> 2;
    while(blkCnt)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        in=arm_nn_read_q7x4_ia(&pIn);
        in=__SSUB8(pad,in);
        in=__SADD8(in,baseV);

        in1 = __SXTB16(__ROR(in, 8));

        /* extend remaining two q7_t values to q15_t values */
        in2 = __SXTB16(in);

#ifndef ARM_MATH_BIG_ENDIAN
        out2 = __PKHTB(in1, in2, 16);
        out1 = __PKHBT(in2, in1, 16);
#else
        out1 = __PKHTB(in1, in2, 16);
        out2 = __PKHBT(in2, in1, 16);
#endif

        shiftV = __USAT16(out1,5);
        *p_out++ = (q7_t) __SSAT((output_base >> (shiftV & 0x0FF)), 8);
        *p_out++ = (q7_t) __SSAT((output_base >> ((shiftV >> 16) & 0x0FF)), 8);

        shiftV = __USAT16(out2,5);
        *p_out++ = (q7_t) __SSAT((output_base >> (shiftV & 0x0FF)), 8);
        *p_out++ = (q7_t) __SSAT((output_base >> ((shiftV >> 16) & 0x0FF)), 8);

        blkCnt --;
    }


    blkCnt = dim_vec & 3;
    while(blkCnt)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__USAT(13 + base - *pIn++, 5);
        *p_out++ = (q7_t) __SSAT((output_base >> shift), 8);

        blkCnt --;
    }
#else
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q15_t     base;

    base = -128;

    /* We first search for the maximum */

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }


    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - Q7BITS;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        shift = (uint8_t)__USAT(vec_in[i] - base, LOG2Q7BITS);
        sum += 0x1 << shift;
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;


    for (i = 0; i < dim_vec; i++)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__USAT(13 + base - vec_in[i], 5);
        p_out[i] = (q7_t) __SSAT((output_base >> shift), 8);

    }
#endif
}