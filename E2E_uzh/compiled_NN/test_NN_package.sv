 
`timescale 1ns / 1ps 

package test_NN_package;

import NVP_v1_constants::*;
import test_package::*;

    localparam SDK_FILE_NAME                = "test_neural_net";
    localparam BASE_DIRECTORY               = "/home/hasan/NVP_v1/tb/compiled_NN/";
    localparam AXI_BYTE_ACCESS_BITS         = 3;
    localparam OUTPUT_LINE_0_START_ADDRESS  = 3072;
    localparam OUTPUT_LINE_1_START_ADDRESS  = 4096;
    localparam OUTPUT_LINE_2_START_ADDRESS  = 5120;

    localparam ACTIVATION_ROWS              = 128; 
    localparam ACTIVATION_COLS              = 128;
    localparam ACTIVATION_CHANNELS          = 8;
    

    localparam LAYER_0_NUMBER_OF_KERNELS                            = 8;            
    localparam LAYER_0_STRIDED_CONV                                 = 1;    
    localparam LAYER_0_BIAS_ENABLE                                  = 1;    
    localparam LAYER_0_RELU_ENABLE                                  = 1;    
    localparam LAYER_0_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_0_Q_SCALE                                      = 1279;    
    localparam LAYER_0_KERNEL_STEPS                                 = 1;    
    localparam LAYER_0_CHANNEL_STEPS                                = 1;        
    localparam LAYER_0_OUTPUT_SLICES                                = 1;        
    localparam LAYER_0_MAX_KERNELS_AND_PES                          = 16;            
    localparam LAYER_0_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 65;                            
    localparam LAYER_0_NUMBER_OF_OUTPUT_COLS                        = 64;                
    localparam LAYER_0_NUMBER_OF_OUTPUT_CH                          = 8;            
    localparam LAYER_0_NUMBER_OF_OUTPUT_ROWS                        = 64;                
    localparam LAYER_0_KERNEL_K                                     = 3;
    localparam LAYER_0_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 48;
    localparam LAYER_0_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 4;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (0),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (ACTIVATION_COLS),            
        .INPUT_NUMBER_OF_CH                  (ACTIVATION_CHANNELS),        
        .INPUT_NUMBER_OF_ROWS                (ACTIVATION_ROWS),            
        .STRIDED_CONV                        (LAYER_0_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_0_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_0_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_0_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_0_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_0_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_0_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_0_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_0_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_0_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_0_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_0_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_0_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_0_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_0;
            
    localparam LAYER_1_NUMBER_OF_KERNELS                            = 16;            
    localparam LAYER_1_STRIDED_CONV                                 = 0;    
    localparam LAYER_1_BIAS_ENABLE                                  = 1;    
    localparam LAYER_1_RELU_ENABLE                                  = 1;    
    localparam LAYER_1_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_1_Q_SCALE                                      = 527;    
    localparam LAYER_1_KERNEL_STEPS                                 = 1;    
    localparam LAYER_1_CHANNEL_STEPS                                = 1;        
    localparam LAYER_1_OUTPUT_SLICES                                = 2;        
    localparam LAYER_1_MAX_KERNELS_AND_PES                          = 16;            
    localparam LAYER_1_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_1_NUMBER_OF_OUTPUT_COLS                        = 64;                
    localparam LAYER_1_NUMBER_OF_OUTPUT_CH                          = 16;            
    localparam LAYER_1_NUMBER_OF_OUTPUT_ROWS                        = 64;                
    localparam LAYER_1_KERNEL_K                                     = 3;
    localparam LAYER_1_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 48;
    localparam LAYER_1_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 8;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (1),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_0_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_0_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_0_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_1_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_1_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_1_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_1_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_1_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_1_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_1_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_1_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_1_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_1_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_1_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_1_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_1_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_1_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_1;
            
    localparam LAYER_2_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_2_STRIDED_CONV                                 = 1;    
    localparam LAYER_2_BIAS_ENABLE                                  = 1;    
    localparam LAYER_2_RELU_ENABLE                                  = 1;    
    localparam LAYER_2_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_2_Q_SCALE                                      = 588;    
    localparam LAYER_2_KERNEL_STEPS                                 = 2;    
    localparam LAYER_2_CHANNEL_STEPS                                = 1;        
    localparam LAYER_2_OUTPUT_SLICES                                = 2;        
    localparam LAYER_2_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_2_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 65;                            
    localparam LAYER_2_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_2_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_2_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_2_KERNEL_K                                     = 3;
    localparam LAYER_2_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 192;
    localparam LAYER_2_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (2),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_1_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_1_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_1_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_2_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_2_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_2_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_2_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_2_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_2_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_2_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_2_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_2_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_2_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_2_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_2_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_2_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_2_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_2;
            
    localparam LAYER_3_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_3_STRIDED_CONV                                 = 0;    
    localparam LAYER_3_BIAS_ENABLE                                  = 1;    
    localparam LAYER_3_RELU_ENABLE                                  = 1;    
    localparam LAYER_3_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_3_Q_SCALE                                      = 517;    
    localparam LAYER_3_KERNEL_STEPS                                 = 2;    
    localparam LAYER_3_CHANNEL_STEPS                                = 2;        
    localparam LAYER_3_OUTPUT_SLICES                                = 2;        
    localparam LAYER_3_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_3_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_3_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_3_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_3_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_3_KERNEL_K                                     = 3;
    localparam LAYER_3_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_3_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (3),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_2_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_2_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_2_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_3_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_3_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_3_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_3_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_3_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_3_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_3_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_3_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_3_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_3_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_3_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_3_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_3_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_3_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_3;
            
    localparam LAYER_4_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_4_STRIDED_CONV                                 = 0;    
    localparam LAYER_4_BIAS_ENABLE                                  = 1;    
    localparam LAYER_4_RELU_ENABLE                                  = 1;    
    localparam LAYER_4_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_4_Q_SCALE                                      = 695;    
    localparam LAYER_4_KERNEL_STEPS                                 = 2;    
    localparam LAYER_4_CHANNEL_STEPS                                = 2;        
    localparam LAYER_4_OUTPUT_SLICES                                = 2;        
    localparam LAYER_4_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_4_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_4_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_4_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_4_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_4_KERNEL_K                                     = 3;
    localparam LAYER_4_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_4_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (4),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_3_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_3_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_3_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_4_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_4_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_4_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_4_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_4_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_4_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_4_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_4_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_4_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_4_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_4_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_4_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_4_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_4_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_4;
            
    localparam LAYER_5_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_5_STRIDED_CONV                                 = 0;    
    localparam LAYER_5_BIAS_ENABLE                                  = 1;    
    localparam LAYER_5_RELU_ENABLE                                  = 1;    
    localparam LAYER_5_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_5_Q_SCALE                                      = 422;    
    localparam LAYER_5_KERNEL_STEPS                                 = 2;    
    localparam LAYER_5_CHANNEL_STEPS                                = 2;        
    localparam LAYER_5_OUTPUT_SLICES                                = 2;        
    localparam LAYER_5_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_5_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_5_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_5_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_5_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_5_KERNEL_K                                     = 3;
    localparam LAYER_5_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_5_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (5),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_4_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_4_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_4_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_5_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_5_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_5_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_5_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_5_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_5_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_5_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_5_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_5_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_5_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_5_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_5_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_5_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_5_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_5;
            
    localparam LAYER_6_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_6_STRIDED_CONV                                 = 0;    
    localparam LAYER_6_BIAS_ENABLE                                  = 1;    
    localparam LAYER_6_RELU_ENABLE                                  = 1;    
    localparam LAYER_6_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_6_Q_SCALE                                      = 387;    
    localparam LAYER_6_KERNEL_STEPS                                 = 2;    
    localparam LAYER_6_CHANNEL_STEPS                                = 2;        
    localparam LAYER_6_OUTPUT_SLICES                                = 2;        
    localparam LAYER_6_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_6_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_6_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_6_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_6_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_6_KERNEL_K                                     = 3;
    localparam LAYER_6_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_6_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (6),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_5_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_5_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_5_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_6_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_6_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_6_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_6_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_6_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_6_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_6_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_6_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_6_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_6_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_6_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_6_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_6_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_6_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_6;
            
    localparam LAYER_7_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_7_STRIDED_CONV                                 = 0;    
    localparam LAYER_7_BIAS_ENABLE                                  = 1;    
    localparam LAYER_7_RELU_ENABLE                                  = 1;    
    localparam LAYER_7_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_7_Q_SCALE                                      = 689;    
    localparam LAYER_7_KERNEL_STEPS                                 = 2;    
    localparam LAYER_7_CHANNEL_STEPS                                = 2;        
    localparam LAYER_7_OUTPUT_SLICES                                = 2;        
    localparam LAYER_7_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_7_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_7_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_7_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_7_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_7_KERNEL_K                                     = 3;
    localparam LAYER_7_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_7_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (7),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_6_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_6_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_6_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_7_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_7_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_7_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_7_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_7_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_7_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_7_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_7_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_7_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_7_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_7_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_7_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_7_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_7_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_7;
            
    localparam LAYER_8_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_8_STRIDED_CONV                                 = 0;    
    localparam LAYER_8_BIAS_ENABLE                                  = 1;    
    localparam LAYER_8_RELU_ENABLE                                  = 1;    
    localparam LAYER_8_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_8_Q_SCALE                                      = 399;    
    localparam LAYER_8_KERNEL_STEPS                                 = 2;    
    localparam LAYER_8_CHANNEL_STEPS                                = 2;        
    localparam LAYER_8_OUTPUT_SLICES                                = 2;        
    localparam LAYER_8_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_8_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_8_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_8_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_8_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_8_KERNEL_K                                     = 3;
    localparam LAYER_8_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_8_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (8),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_7_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_7_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_7_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_8_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_8_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_8_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_8_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_8_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_8_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_8_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_8_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_8_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_8_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_8_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_8_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_8_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_8_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_8;
            
    localparam LAYER_9_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_9_STRIDED_CONV                                 = 0;    
    localparam LAYER_9_BIAS_ENABLE                                  = 1;    
    localparam LAYER_9_RELU_ENABLE                                  = 1;    
    localparam LAYER_9_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_9_Q_SCALE                                      = 325;    
    localparam LAYER_9_KERNEL_STEPS                                 = 2;    
    localparam LAYER_9_CHANNEL_STEPS                                = 2;        
    localparam LAYER_9_OUTPUT_SLICES                                = 2;        
    localparam LAYER_9_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_9_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_9_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_9_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_9_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_9_KERNEL_K                                     = 3;
    localparam LAYER_9_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_9_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (9),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_8_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_8_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_8_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_9_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_9_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_9_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_9_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_9_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_9_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_9_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_9_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_9_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_9_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_9_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_9_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_9_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_9_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_9;
            
    localparam LAYER_10_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_10_STRIDED_CONV                                 = 0;    
    localparam LAYER_10_BIAS_ENABLE                                  = 1;    
    localparam LAYER_10_RELU_ENABLE                                  = 1;    
    localparam LAYER_10_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_10_Q_SCALE                                      = 327;    
    localparam LAYER_10_KERNEL_STEPS                                 = 2;    
    localparam LAYER_10_CHANNEL_STEPS                                = 2;        
    localparam LAYER_10_OUTPUT_SLICES                                = 2;        
    localparam LAYER_10_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_10_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_10_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_10_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_10_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_10_KERNEL_K                                     = 3;
    localparam LAYER_10_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_10_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (10),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_9_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_9_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_9_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_10_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_10_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_10_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_10_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_10_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_10_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_10_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_10_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_10_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_10_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_10_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_10_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_10_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_10_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_10;
            
    localparam LAYER_11_NUMBER_OF_KERNELS                            = 32;            
    localparam LAYER_11_STRIDED_CONV                                 = 0;    
    localparam LAYER_11_BIAS_ENABLE                                  = 1;    
    localparam LAYER_11_RELU_ENABLE                                  = 1;    
    localparam LAYER_11_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_11_Q_SCALE                                      = 429;    
    localparam LAYER_11_KERNEL_STEPS                                 = 2;    
    localparam LAYER_11_CHANNEL_STEPS                                = 2;        
    localparam LAYER_11_OUTPUT_SLICES                                = 2;        
    localparam LAYER_11_MAX_KERNELS_AND_PES                          = 32;            
    localparam LAYER_11_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 64;                            
    localparam LAYER_11_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_11_NUMBER_OF_OUTPUT_CH                          = 32;            
    localparam LAYER_11_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_11_KERNEL_K                                     = 3;
    localparam LAYER_11_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 384;
    localparam LAYER_11_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 16;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (11),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_10_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_10_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_10_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_11_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_11_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_11_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_11_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_11_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_11_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_11_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_11_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_11_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_11_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_11_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_11_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_11_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_11_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_11;
            
    localparam LAYER_12_NUMBER_OF_KERNELS                            = 16;            
    localparam LAYER_12_STRIDED_CONV                                 = 0;    
    localparam LAYER_12_BIAS_ENABLE                                  = 1;    
    localparam LAYER_12_RELU_ENABLE                                  = 1;    
    localparam LAYER_12_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_12_Q_SCALE                                      = 486;    
    localparam LAYER_12_KERNEL_STEPS                                 = 1;    
    localparam LAYER_12_CHANNEL_STEPS                                = 2;        
    localparam LAYER_12_OUTPUT_SLICES                                = 2;        
    localparam LAYER_12_MAX_KERNELS_AND_PES                          = 16;            
    localparam LAYER_12_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 32;                            
    localparam LAYER_12_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_12_NUMBER_OF_OUTPUT_CH                          = 16;            
    localparam LAYER_12_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_12_KERNEL_K                                     = 3;
    localparam LAYER_12_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 192;
    localparam LAYER_12_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 8;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (12),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_11_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_11_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_11_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_12_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_12_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_12_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_12_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_12_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_12_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_12_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_12_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_12_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_12_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_12_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_12_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_12_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_12_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_12;
            
    localparam LAYER_13_NUMBER_OF_KERNELS                            = 1;            
    localparam LAYER_13_STRIDED_CONV                                 = 0;    
    localparam LAYER_13_BIAS_ENABLE                                  = 1;    
    localparam LAYER_13_RELU_ENABLE                                  = 0;    
    localparam LAYER_13_COMPRESS_OUTPUT                              = 1;    
    localparam LAYER_13_Q_SCALE                                      = 223;    
    localparam LAYER_13_KERNEL_STEPS                                 = 1;    
    localparam LAYER_13_CHANNEL_STEPS                                = 1;        
    localparam LAYER_13_OUTPUT_SLICES                                = 1;        
    localparam LAYER_13_MAX_KERNELS_AND_PES                          = 16;            
    localparam LAYER_13_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = 32;                            
    localparam LAYER_13_NUMBER_OF_OUTPUT_COLS                        = 32;                
    localparam LAYER_13_NUMBER_OF_OUTPUT_CH                          = 1;            
    localparam LAYER_13_NUMBER_OF_OUTPUT_ROWS                        = 32;                
    localparam LAYER_13_KERNEL_K                                     = 3;
    localparam LAYER_13_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = 96;
    localparam LAYER_13_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = 1;                                        
        
    neural_network_layer #(
        .LAYER_ID                            (13),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_12_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_12_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_12_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_13_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_13_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_13_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_13_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_13_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_13_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_13_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_13_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_13_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_13_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_13_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_13_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_13_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_13_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_13;
             

endpackage
    
