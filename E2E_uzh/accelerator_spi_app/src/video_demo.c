/************************************************************************/
/*																		*/
/*	video_demo.c	--	ZYBO Video demonstration 						*/
/*																		*/
/************************************************************************/
/*	Author: Sam Bobrowicz												*/
/*	Copyright 2015, Digilent Inc.										*/
/************************************************************************/
/*  Module Description: 												*/
/*																		*/
/*		This file contains code for running a demonstration of the		*/
/*		Video input and output capabilities on the ZYBO. It is a good	*/
/*		example of how to properly use the display_ctrl and				*/
/*		video_capture drivers.											*/
/*																		*/
/*																		*/
/************************************************************************/
/*  Revision History:													*/
/* 																		*/
/*		11/25/2015(SamB): Created										*/
/*																		*/
/************************************************************************/

/* ------------------------------------------------------------ */
/*				Include File Definitions						*/
/* ------------------------------------------------------------ */
#include <stdio.h>
#include "xuartps.h"
#include "math.h"
#include <ctype.h>
#include <stdlib.h>
#include "xil_cache.h"
#include "xparameters.h"
#include "platform.h"
#include "xil_cache.h"
#include "xaxicdma.h"
#include "xil_types.h"
#include "xil_exception.h"
#include "video_demo.h"
#include "test_neural_net.h"
#include "test_images.h"
#include "test_compressed_images.h"
//#include "neural_net_layer.h"
#include "xgpiops.h"
#include "imec_spi.h"

/*
 * XPAR redefines
 */
#define DYNCLK_BASEADDR 		XPAR_AXI_DYNCLK_0_S_AXI_LITE_BASEADDR
#define VDMA_ID 				XPAR_AXIVDMA_0_DEVICE_ID
#define HDMI_OUT_VTC_ID 		XPAR_V_TC_OUT_DEVICE_ID
#define HDMI_IN_VTC_ID 			XPAR_V_TC_IN_DEVICE_ID
#define HDMI_IN_GPIO_ID 		XPAR_AXI_GPIO_VIDEO_DEVICE_ID
#define HDMI_IN_VTC_IRPT_ID 	XPAR_FABRIC_V_TC_IN_IRQ_INTR
#define HDMI_IN_GPIO_IRPT_ID 	XPAR_FABRIC_AXI_GPIO_VIDEO_IP2INTC_IRPT_INTR
#define SCU_TIMER_ID 			XPAR_SCUTIMER_DEVICE_ID
#define UART_BASEADDR 			XPAR_PS7_UART_1_BASEADDR
#define request_pin 	13 // ZYBO PMOD JF1
#define request_pin_led 10 // ZYBO PMOD JF2
#define ack_pin 		0  // ZYBO PMOD JF7
#define ack_pin_led 	9  // ZYBO PMOD JF8


// PACKAGE_PIN V12 -> SPI1_SS_O   -> ZYBO PMOD JE1			 
// PACKAGE_PIN W16 -> SPI1_MOSI_O -> ZYBO PMOD JE2			 
// PACKAGE_PIN J15 -> SPI1_MISO_I -> ZYBO PMOD JE3			         
// PACKAGE_PIN H15 -> SPI1_SCLK_O -> ZYBO PMOD JE4

/* ------------------------------------------------------------ */
/*				Global Variables								*/
/* ------------------------------------------------------------ */
XGpioPs Gpio;

/*
 * Display and Video Driver structs
 */
DisplayCtrl dispCtrl;
XAxiVdma vdma;
VideoCapture videoCapt;
INTC intc;
char fRefresh; //flag used to trigger a refresh of the Menu on video detect
int interrupts_enabled = 0;
/*
 * Framebuffers for video data
 */
u8 frameBuf[DISPLAY_NUM_FRAMES][DEMO_MAX_FRAME] __attribute__((aligned(0x20)));
u8 *pFrames[DISPLAY_NUM_FRAMES]; //array of pointers to the frame buffers

/*
 * Interrupt vector table
 */
const ivt_t ivt[] = {
	videoGpioIvt(HDMI_IN_GPIO_IRPT_ID, &videoCapt),
	videoVtcIvt(HDMI_IN_VTC_IRPT_ID, &(videoCapt.vtc))
};

/* ------------------------------------------------------------ */
/*				Procedure Definitions							*/
/* ------------------------------------------------------------ */

void run_NN(NN_config_t *NN, u8 *srcFrame, uint64_t *compressed_src_frame, u8 *destFrame, u32 width, u32 height, u32 stride, u32 start_pixel_width, u32 start_pixel_height);

int main(void)
{
//	init_platform();
	init_platform();
	print("-----------------------------------------\n\r");
	print("Hello World \n\r");
	int Status;

	// CDMA init
	XAxiCdma axi_cdma_0_instance;
	init_dma(&axi_cdma_0_instance);

	// SPI init
	if (XST_SUCCESS != spi_master_init()) {
		return XST_FAILURE;
	}

	// INTERRUPT Controller init
	Status = fnInitInterruptController(&intc);
	if(Status != XST_SUCCESS) {
		xil_printf("Error initializing interrupt controller");
		return XST_FAILURE;
	}
	fnEnableInterrupts(&intc, &ivt[0], sizeof(ivt)/sizeof(ivt[0]));
	interrupts_enabled = 1;

	// GPIO init
	XGpio_Config *GPIOConfigPtr = XGpioPs_LookupConfig(XPAR_PS7_GPIO_0_DEVICE_ID);
	if(XGpioPs_CfgInitialize(&Gpio, GPIOConfigPtr, GPIOConfigPtr->BaseAddress) != XST_SUCCESS){
		xil_printf("Error initializing GPIO\r\n");
			return XST_FAILURE;
	}
	// request_pin
	// request_pin_led
	// ack_pin
	// ack_pin_led

	XSpiPs_SetSlaveSelect(&spi, 1);

	// request 
	XGpioPs_SetDirectionPin(&Gpio, request_pin, 1);
	XGpioPs_SetDirectionPin(&Gpio, request_pin_led, 1);
	XGpioPs_SetOutputEnablePin(&Gpio, request_pin, 1);
	XGpioPs_SetOutputEnablePin(&Gpio, request_pin_led, 1);
	XGpioPs_WritePin(&Gpio, request_pin, 0x0);
	XGpioPs_WritePin(&Gpio, request_pin_led, 0x0);
	sleep(1);
	XGpioPs_WritePin(&Gpio, request_pin, 0x1);
	XGpioPs_WritePin(&Gpio, request_pin_led, 0x1);
	sleep(1);

	// ack
	XGpioPs_SetDirectionPin(&Gpio, ack_pin, 0);
	XGpioPs_SetDirectionPin(&Gpio, ack_pin_led, 1);
	XGpioPs_SetOutputEnablePin(&Gpio, ack_pin_led, 1);
	u32 ack = 0;
	XGpioPs_WritePin(&Gpio, ack_pin_led, ack);
	while (ack==0){
		ack = XGpioPs_ReadPin(&Gpio, ack_pin);
		XGpioPs_WritePin(&Gpio, ack_pin_led, ack);
	}
	xil_printf("Finished handshake. \r\n");

	// SPI test
	if (XST_SUCCESS != imec_test_configuation()) {
		xil_printf("SPI test FAILED\r\n");
		return XST_FAILURE;
	} else {
		ack = XGpioPs_ReadPin(&Gpio, ack_pin);
		XGpioPs_WritePin(&Gpio, ack_pin_led, ack);
		if (XST_SUCCESS != imec_test_configuation_set_pattern(2)) {
			xil_printf("SPI test FAILED\r\n");
			return XST_FAILURE;
		}
		else {
			xil_printf("SPI test PASSED\r\n");
		}
		
	}

	unsigned char os_cfg_reg[8][16] = {0};
	//col 1
	os_cfg_reg[0][15] = 1;
	os_cfg_reg[4][15] = 1;
	os_cfg_reg[0][8]  = 1;
	os_cfg_reg[4][8]  = 1;
	os_cfg_reg[0][1]  = 1;
	os_cfg_reg[4][1]  = 1;
	os_cfg_reg[1][10] = 1;
	os_cfg_reg[5][10] = 1;
	os_cfg_reg[1][3]  = 1;
	os_cfg_reg[5][3]  = 1;
	os_cfg_reg[2][10] = 1;
	os_cfg_reg[6][10] = 1;
	os_cfg_reg[3][11] = 1;
	os_cfg_reg[7][11] = 1;

	//col 2
	os_cfg_reg[0][14] = 0;
	os_cfg_reg[4][14] = 0;
	os_cfg_reg[0][7]  = 0;
	os_cfg_reg[4][7]  = 0;
	os_cfg_reg[0][0]  = 0;
	os_cfg_reg[4][0]  = 0;
	os_cfg_reg[1][9]  = 0;
	os_cfg_reg[5][9]  = 0;
	os_cfg_reg[1][2]  = 0;
	os_cfg_reg[5][2]  = 0;
	os_cfg_reg[2][9]  = 0;
	os_cfg_reg[6][9]  = 0;
	os_cfg_reg[3][10] = 0;
	os_cfg_reg[7][10] = 0;

	//col 3
	os_cfg_reg[0][13] = 1;
	os_cfg_reg[4][13] = 1;
	os_cfg_reg[0][6]  = 1;
	os_cfg_reg[4][6]  = 1;
	os_cfg_reg[1][15] = 1;
	os_cfg_reg[5][15] = 1;
	os_cfg_reg[1][8]  = 1;
	os_cfg_reg[5][8]  = 1;
	os_cfg_reg[1][1]  = 1;
	os_cfg_reg[5][1]  = 1;
	os_cfg_reg[2][8]  = 1;
	os_cfg_reg[6][8]  = 1;
	os_cfg_reg[3][9]  = 1;
	os_cfg_reg[7][9]  = 1;

	//col 4
	os_cfg_reg[0][12] = 0;
	os_cfg_reg[4][12] = 0;
	os_cfg_reg[0][5]  = 0;
	os_cfg_reg[4][5]  = 0;
	os_cfg_reg[1][14] = 0;
	os_cfg_reg[5][14] = 0;
	os_cfg_reg[1][7]  = 0;
	os_cfg_reg[5][7]  = 0;
	os_cfg_reg[1][0]  = 0;
	os_cfg_reg[5][0]  = 0;
	os_cfg_reg[2][7]  = 0;
	os_cfg_reg[6][7]  = 0;
	os_cfg_reg[3][8]  = 0;
	os_cfg_reg[7][8]  = 0;

	//col 5
	os_cfg_reg[0][11] = 0;
	os_cfg_reg[4][11] = 0;
	os_cfg_reg[0][4]  = 0;
	os_cfg_reg[4][4]  = 0;
	os_cfg_reg[1][13] = 0;
	os_cfg_reg[5][13] = 0;
	os_cfg_reg[1][6]  = 0;
	os_cfg_reg[5][6]  = 0;
	os_cfg_reg[2][13] = 0;
	os_cfg_reg[6][13] = 0;
	os_cfg_reg[2][6]  = 0;
	os_cfg_reg[6][6]  = 0;
	os_cfg_reg[3][7]  = 0;
	os_cfg_reg[7][7]  = 0;

	//col 6
	os_cfg_reg[0][10] = 1;
	os_cfg_reg[4][10] = 1;
	os_cfg_reg[0][3]  = 1;
	os_cfg_reg[4][3]  = 1;
	os_cfg_reg[1][12] = 1;
	os_cfg_reg[5][12] = 1;
	os_cfg_reg[1][5]  = 1;
	os_cfg_reg[5][5]  = 1;
	os_cfg_reg[2][12] = 1;
	os_cfg_reg[6][12] = 1;
	os_cfg_reg[2][5]  = 1;
	os_cfg_reg[6][5]  = 1;
	os_cfg_reg[3][6]  = 1;
	os_cfg_reg[7][6]  = 1;

	//col 7
	os_cfg_reg[0][9]  = 1;
	os_cfg_reg[4][9]  = 1;
	os_cfg_reg[0][2]  = 1;
	os_cfg_reg[4][2]  = 1;
	os_cfg_reg[1][11] = 1;
	os_cfg_reg[5][11] = 1;
	os_cfg_reg[1][4]  = 1;
	os_cfg_reg[5][4]  = 1;
	os_cfg_reg[2][11] = 1;
	os_cfg_reg[6][11] = 1;
	os_cfg_reg[2][4]  = 1;
	os_cfg_reg[6][4]  = 1;
	os_cfg_reg[3][5]  = 1;
	os_cfg_reg[7][5]  = 1;
































	

	//col 1
	// os_cfg_reg[0][15] = 1;
	// os_cfg_reg[0][14] = 1;
	// os_cfg_reg[0][13] = 1;
	// os_cfg_reg[0][12] = 1;
	// os_cfg_reg[0][11] = 1;
	// os_cfg_reg[0][10] = 1;
	// os_cfg_reg[0][9]  = 1;

	// //col 3
	// os_cfg_reg[0][8] = 0;
	// os_cfg_reg[0][7] = 0;
	// os_cfg_reg[0][6] = 0;
	// os_cfg_reg[0][5] = 0;
	// os_cfg_reg[0][4] = 0;
	// os_cfg_reg[0][3] = 0;
	// os_cfg_reg[0][2] = 0;

	// //col 5
	// os_cfg_reg[0][1]  = 1;
	// os_cfg_reg[0][0]  = 1;
	// os_cfg_reg[1][15] = 1;
	// os_cfg_reg[1][14] = 1;
	// os_cfg_reg[1][13] = 1;
	// os_cfg_reg[1][12] = 1;
	// os_cfg_reg[1][11] = 1;

	// //col 7
	// os_cfg_reg[1][10] = 0;
	// os_cfg_reg[1][9]  = 0;
	// os_cfg_reg[1][8]  = 0;
	// os_cfg_reg[1][7]  = 0;
	// os_cfg_reg[1][6]  = 0;
	// os_cfg_reg[1][5]  = 0;
	// os_cfg_reg[1][4]  = 0;

	// //col 9
	// os_cfg_reg[1][3]  = 0;
	// os_cfg_reg[1][2]  = 0;
	// os_cfg_reg[1][1]  = 0;
	// os_cfg_reg[1][0]  = 0;
	// os_cfg_reg[2][13] = 0;
	// os_cfg_reg[2][12] = 0;
	// os_cfg_reg[2][11] = 0;

	// //col 11
	// os_cfg_reg[2][10] = 1;
	// os_cfg_reg[2][9]  = 1;
	// os_cfg_reg[2][8]  = 1;
	// os_cfg_reg[2][7]  = 1;
	// os_cfg_reg[2][6]  = 1;
	// os_cfg_reg[2][5]  = 1;
	// os_cfg_reg[2][4]  = 1;

	// //col 13
	// os_cfg_reg[3][11] = 1;
	// os_cfg_reg[3][10] = 1;
	// os_cfg_reg[3][9]  = 1;
	// os_cfg_reg[3][8]  = 1;
	// os_cfg_reg[3][7]  = 1;
	// os_cfg_reg[3][6]  = 1;
	// os_cfg_reg[3][5]  = 1;

	// //col 2
	// os_cfg_reg[4][15] = 1;
	// os_cfg_reg[4][14] = 1;
	// os_cfg_reg[4][13] = 1;
	// os_cfg_reg[4][12] = 1;
	// os_cfg_reg[4][11] = 1;
	// os_cfg_reg[4][10] = 1;
	// os_cfg_reg[4][9]  = 1;

	// //col 4
	// os_cfg_reg[4][8] = 0;
	// os_cfg_reg[4][7] = 0;
	// os_cfg_reg[4][6] = 0;
	// os_cfg_reg[4][5] = 0;
	// os_cfg_reg[4][4] = 0;
	// os_cfg_reg[4][3] = 0;
	// os_cfg_reg[4][2] = 0;

	// //col 6
	// os_cfg_reg[4][1]  = 0;
	// os_cfg_reg[4][0]  = 0;
	// os_cfg_reg[5][15] = 0;
	// os_cfg_reg[5][14] = 0;
	// os_cfg_reg[5][13] = 0;
	// os_cfg_reg[5][12] = 0;
	// os_cfg_reg[5][11] = 0;

	// //col 8
	// os_cfg_reg[5][10] = 0;
	// os_cfg_reg[5][9]  = 0;
	// os_cfg_reg[5][8]  = 0;
	// os_cfg_reg[5][7]  = 0;
	// os_cfg_reg[5][6]  = 0;
	// os_cfg_reg[5][5]  = 0;
	// os_cfg_reg[5][4]  = 0;

	// //col 10
	// os_cfg_reg[5][3]  = 1;
	// os_cfg_reg[5][2]  = 1;
	// os_cfg_reg[5][1]  = 1;
	// os_cfg_reg[5][0]  = 1;
	// os_cfg_reg[6][13] = 1;
	// os_cfg_reg[6][12] = 1;
	// os_cfg_reg[6][11] = 1;

	// //col 12
	// os_cfg_reg[6][10] = 0;
	// os_cfg_reg[6][9]  = 0;
	// os_cfg_reg[6][8]  = 0;
	// os_cfg_reg[6][7]  = 0;
	// os_cfg_reg[6][6]  = 0;
	// os_cfg_reg[6][5]  = 0;
	// os_cfg_reg[6][4]  = 0;

	// //col 14
	// os_cfg_reg[7][11] = 1;
	// os_cfg_reg[7][10] = 1;
	// os_cfg_reg[7][9]  = 1;
	// os_cfg_reg[7][8]  = 1;
	// os_cfg_reg[7][7]  = 1;
	// os_cfg_reg[7][6]  = 1;
	// os_cfg_reg[7][5]  = 1;	

	imec_phosphene_14x7_output(os_cfg_reg);


	// while(1){
	// 	ack = XGpioPs_ReadPin(&Gpio, ack_pin);
	// 	XGpioPs_WritePin(&Gpio, ack_pin_led, ack);
	// }
	// return;


	// NN definition
	uint16_t verbose = 0;
	uint64_t *layer_0_output_ground_truth   = layer_0_ground_truth_output_activations;
	uint64_t *layer_1_output_ground_truth   = layer_1_ground_truth_output_activations;
	uint64_t *layer_2_output_ground_truth   = layer_2_ground_truth_output_activations;
	uint64_t *layer_3_output_ground_truth   = layer_3_ground_truth_output_activations;
	uint64_t *layer_4_output_ground_truth   = layer_4_ground_truth_output_activations;
	uint64_t *layer_5_output_ground_truth   = layer_5_ground_truth_output_activations;
	uint64_t *layer_6_output_ground_truth   = layer_6_ground_truth_output_activations;
	uint64_t *layer_7_output_ground_truth   = layer_7_ground_truth_output_activations;
	uint64_t *layer_8_output_ground_truth   = layer_8_ground_truth_output_activations;
	uint64_t *layer_9_output_ground_truth   = layer_9_ground_truth_output_activations;
	uint64_t *layer_10_output_ground_truth   = layer_10_ground_truth_output_activations;
	uint64_t *layer_11_output_ground_truth   = layer_11_ground_truth_output_activations;
	uint64_t *layer_12_output_ground_truth   = layer_12_ground_truth_output_activations;
	uint64_t *layer_13_output_ground_truth   = layer_13_ground_truth_output_activations;

	uint64_t *layer_0_output_image;
	uint64_t *layer_1_output_image;
	uint64_t *layer_2_output_image;
	uint64_t *layer_3_output_image;
	uint64_t *layer_4_output_image;
	uint64_t *layer_5_output_image;
	uint64_t *layer_6_output_image;
	uint64_t *layer_7_output_image;
	uint64_t *layer_8_output_image;
	uint64_t *layer_9_output_image;
	uint64_t *layer_10_output_image;
	uint64_t *layer_11_output_image;
	uint64_t *layer_12_output_image;
	uint64_t *layer_13_output_image;

	layer_0_output_image = malloc(layer_0_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_1_output_image = malloc(layer_1_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_2_output_image = malloc(layer_2_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_3_output_image = malloc(layer_3_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_4_output_image = malloc(layer_4_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_5_output_image = malloc(layer_5_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_6_output_image = malloc(layer_6_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_7_output_image = malloc(layer_7_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_8_output_image = malloc(layer_8_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_9_output_image = malloc(layer_9_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_10_output_image = malloc(layer_10_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_11_output_image = malloc(layer_11_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_12_output_image = malloc(layer_12_total_number_of_output_activations_entries*sizeof(uint64_t));
	layer_13_output_image = malloc(layer_13_total_number_of_output_activations_entries*sizeof(uint64_t));
	Xil_DCacheFlush();

	layer_config_t layer_0 = {
		.number_of_entries_per_row					=	number_of_entries_per_row,
		.total_number_of_input_activations_entries	=	total_number_of_input_activations_entries,
//		.input_activations							=	input_activations,
		.input_activations							=	compressed_test_image[0],
		.input_cols									=	LAYER_0_INPUT_COLS,
		.input_rows									=	LAYER_0_INPUT_ROWS,
		.input_ch									=	LAYER_0_INPUT_CH,
		.kernel_k									=	LAYER_0_KERNEL_K,
		.output_cols								=	LAYER_0_OUTPUT_COLS,
		.output_rows								=	LAYER_0_OUTPUT_ROWS,
		.output_ch									=	LAYER_0_OUTPUT_CH,
		.output_image								= 	layer_0_output_image,
		.number_of_entries_per_output_row			=	layer_0_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_0_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_0_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_0_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_0_weight_array_0,
		.weight_array_1							    =	layer_0_weight_array_1,
		.weight_array_2							    =	layer_0_weight_array_2,
		.number_of_entries_bias_array				= 	layer_0_number_of_entries_bias_array,
		.bias_array									=   layer_0_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_0_pre_reg,
		.intra_reg								    =	&layer_0_intra_reg,
		.strided									=   1,
		.input_layer 								=	1
	};
	layer_config_t layer_1 = {
		.number_of_entries_per_row					=	layer_0_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_0_total_number_of_output_activations_entries,
		.input_activations							=	layer_0_output_image,
		.input_cols									=	LAYER_0_OUTPUT_COLS,
		.input_rows									=	LAYER_0_OUTPUT_ROWS,
		.input_ch									=	LAYER_0_OUTPUT_CH,
		.kernel_k									=	LAYER_1_KERNEL_K,
		.output_cols								=	LAYER_1_OUTPUT_COLS,
		.output_rows								=	LAYER_1_OUTPUT_ROWS,
		.output_ch									=	LAYER_1_OUTPUT_CH,
		.output_image								= 	layer_1_output_image,
		.number_of_entries_per_output_row			=	layer_1_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_1_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_1_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_1_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_1_weight_array_0,
		.weight_array_1							    =	layer_1_weight_array_1,
		.weight_array_2							    =	layer_1_weight_array_2,
		.number_of_entries_bias_array				= 	layer_1_number_of_entries_bias_array,
		.bias_array									=   layer_1_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_1_pre_reg,
		.intra_reg								    =	&layer_1_intra_reg,
		.strided									=   0,
		.input_layer 								=	0
	};
	layer_config_t layer_2 = {
		.number_of_entries_per_row					=	layer_1_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_1_total_number_of_output_activations_entries,
		.input_activations							=	layer_1_output_image,
		.input_cols									=	LAYER_1_OUTPUT_COLS,
		.input_rows									=	LAYER_1_OUTPUT_ROWS,
		.input_ch									=	LAYER_1_OUTPUT_CH,
		.kernel_k									=	LAYER_2_KERNEL_K,
		.output_cols								=	LAYER_2_OUTPUT_COLS,
		.output_rows								=	LAYER_2_OUTPUT_ROWS,
		.output_ch									=	LAYER_2_OUTPUT_CH,
		.output_image								= 	layer_2_output_image,
		.number_of_entries_per_output_row			=	layer_2_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_2_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_2_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_2_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_2_weight_array_0,
		.weight_array_1							    =	layer_2_weight_array_1,
		.weight_array_2							    =	layer_2_weight_array_2,
		.number_of_entries_bias_array				= 	layer_2_number_of_entries_bias_array,
		.bias_array									=   layer_2_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_2_pre_reg,
		.intra_reg								    =	&layer_2_intra_reg,
		.strided									=   1,
		.input_layer 								=	0
	};
	layer_config_t layer_3 = {
		.number_of_entries_per_row					=	layer_2_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_2_total_number_of_output_activations_entries,
		.input_activations							=	layer_2_output_image,
		.input_cols									=	LAYER_2_OUTPUT_COLS,
		.input_rows									=	LAYER_2_OUTPUT_ROWS,
		.input_ch									=	LAYER_2_OUTPUT_CH,
		.kernel_k									=	LAYER_3_KERNEL_K,
		.output_cols								=	LAYER_3_OUTPUT_COLS,
		.output_rows								=	LAYER_3_OUTPUT_ROWS,
		.output_ch									=	LAYER_3_OUTPUT_CH,
		.output_image								= 	layer_3_output_image,
		.number_of_entries_per_output_row			=	layer_3_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_3_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_3_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_3_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_3_weight_array_0,
		.weight_array_1							    =	layer_3_weight_array_1,
		.weight_array_2							    =	layer_3_weight_array_2,
		.number_of_entries_bias_array				= 	layer_3_number_of_entries_bias_array,
		.bias_array									=   layer_3_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_3_pre_reg,
		.intra_reg								    =	&layer_3_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_4 = {
		.number_of_entries_per_row					=	layer_3_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_3_total_number_of_output_activations_entries,
		.input_activations							=	layer_3_output_image,
		.input_cols									=	LAYER_3_OUTPUT_COLS,
		.input_rows									=	LAYER_3_OUTPUT_ROWS,
		.input_ch									=	LAYER_3_OUTPUT_CH,
		.kernel_k									=	LAYER_4_KERNEL_K,
		.output_cols								=	LAYER_4_OUTPUT_COLS,
		.output_rows								=	LAYER_4_OUTPUT_ROWS,
		.output_ch									=	LAYER_4_OUTPUT_CH,
		.output_image								= 	layer_4_output_image,
		.number_of_entries_per_output_row			=	layer_4_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_4_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_4_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_4_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_4_weight_array_0,
		.weight_array_1							    =	layer_4_weight_array_1,
		.weight_array_2							    =	layer_4_weight_array_2,
		.number_of_entries_bias_array				= 	layer_4_number_of_entries_bias_array,
		.bias_array									=   layer_4_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_4_pre_reg,
		.intra_reg								    =	&layer_4_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_5 = {
		.number_of_entries_per_row					=	layer_4_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_4_total_number_of_output_activations_entries,
		.input_activations							=	layer_4_output_image,
		.input_cols									=	LAYER_4_OUTPUT_COLS,
		.input_rows									=	LAYER_4_OUTPUT_ROWS,
		.input_ch									=	LAYER_4_OUTPUT_CH,
		.kernel_k									=	LAYER_5_KERNEL_K,
		.output_cols								=	LAYER_5_OUTPUT_COLS,
		.output_rows								=	LAYER_5_OUTPUT_ROWS,
		.output_ch									=	LAYER_5_OUTPUT_CH,
		.output_image								= 	layer_5_output_image,
		.number_of_entries_per_output_row			=	layer_5_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_5_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_5_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_5_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_5_weight_array_0,
		.weight_array_1							    =	layer_5_weight_array_1,
		.weight_array_2							    =	layer_5_weight_array_2,
		.number_of_entries_bias_array				= 	layer_5_number_of_entries_bias_array,
		.bias_array									=   layer_5_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_5_pre_reg,
		.intra_reg								    =	&layer_5_intra_reg,
		.strided									=   0,
		.input_layer 								=	0
	};
	layer_config_t layer_6 = {
		.number_of_entries_per_row					=	layer_5_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_5_total_number_of_output_activations_entries,
		.input_activations							=	layer_5_output_image,
		.input_cols									=	LAYER_5_OUTPUT_COLS,
		.input_rows									=	LAYER_5_OUTPUT_ROWS,
		.input_ch									=	LAYER_5_OUTPUT_CH,
		.kernel_k									=	LAYER_6_KERNEL_K,
		.output_cols								=	LAYER_6_OUTPUT_COLS,
		.output_rows								=	LAYER_6_OUTPUT_ROWS,
		.output_ch									=	LAYER_6_OUTPUT_CH,
		.output_image								= 	layer_6_output_image,
		.number_of_entries_per_output_row			=	layer_6_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_6_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_6_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_6_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_6_weight_array_0,
		.weight_array_1							    =	layer_6_weight_array_1,
		.weight_array_2							    =	layer_6_weight_array_2,
		.number_of_entries_bias_array				= 	layer_6_number_of_entries_bias_array,
		.bias_array									=   layer_6_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_6_pre_reg,
		.intra_reg								    =	&layer_6_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_7 = {
		.number_of_entries_per_row					=	layer_6_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_6_total_number_of_output_activations_entries,
		.input_activations							=	layer_6_output_image,
		.input_cols									=	LAYER_6_OUTPUT_COLS,
		.input_rows									=	LAYER_6_OUTPUT_ROWS,
		.input_ch									=	LAYER_6_OUTPUT_CH,
		.kernel_k									=	LAYER_7_KERNEL_K,
		.output_cols								=	LAYER_7_OUTPUT_COLS,
		.output_rows								=	LAYER_7_OUTPUT_ROWS,
		.output_ch									=	LAYER_7_OUTPUT_CH,
		.output_image								= 	layer_7_output_image,
		.number_of_entries_per_output_row			=	layer_7_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_7_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_7_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_7_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_7_weight_array_0,
		.weight_array_1							    =	layer_7_weight_array_1,
		.weight_array_2							    =	layer_7_weight_array_2,
		.number_of_entries_bias_array				= 	layer_7_number_of_entries_bias_array,
		.bias_array									=   layer_7_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_7_pre_reg,
		.intra_reg								    =	&layer_7_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_8 = {
		.number_of_entries_per_row					=	layer_7_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_7_total_number_of_output_activations_entries,
		.input_activations							=	layer_7_output_image,
		.input_cols									=	LAYER_7_OUTPUT_COLS,
		.input_rows									=	LAYER_7_OUTPUT_ROWS,
		.input_ch									=	LAYER_7_OUTPUT_CH,
		.kernel_k									=	LAYER_8_KERNEL_K,
		.output_cols								=	LAYER_8_OUTPUT_COLS,
		.output_rows								=	LAYER_8_OUTPUT_ROWS,
		.output_ch									=	LAYER_8_OUTPUT_CH,
		.output_image								= 	layer_8_output_image,
		.number_of_entries_per_output_row			=	layer_8_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_8_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_8_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_8_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_8_weight_array_0,
		.weight_array_1							    =	layer_8_weight_array_1,
		.weight_array_2							    =	layer_8_weight_array_2,
		.number_of_entries_bias_array				= 	layer_8_number_of_entries_bias_array,
		.bias_array									=   layer_8_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_8_pre_reg,
		.intra_reg								    =	&layer_8_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_9 = {
		.number_of_entries_per_row					=	layer_8_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_8_total_number_of_output_activations_entries,
		.input_activations							=	layer_8_output_image,
		.input_cols									=	LAYER_8_OUTPUT_COLS,
		.input_rows									=	LAYER_8_OUTPUT_ROWS,
		.input_ch									=	LAYER_8_OUTPUT_CH,
		.kernel_k									=	LAYER_9_KERNEL_K,
		.output_cols								=	LAYER_9_OUTPUT_COLS,
		.output_rows								=	LAYER_9_OUTPUT_ROWS,
		.output_ch									=	LAYER_9_OUTPUT_CH,
		.output_image								= 	layer_9_output_image,
		.number_of_entries_per_output_row			=	layer_9_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_9_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_9_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_9_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_9_weight_array_0,
		.weight_array_1							    =	layer_9_weight_array_1,
		.weight_array_2							    =	layer_9_weight_array_2,
		.number_of_entries_bias_array				= 	layer_9_number_of_entries_bias_array,
		.bias_array									=   layer_9_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_9_pre_reg,
		.intra_reg								    =	&layer_9_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
		layer_config_t layer_10 = {
		.number_of_entries_per_row					=	layer_9_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_9_total_number_of_output_activations_entries,
		.input_activations							=	layer_9_output_image,
		.input_cols									=	LAYER_9_OUTPUT_COLS,
		.input_rows									=	LAYER_9_OUTPUT_ROWS,
		.input_ch									=	LAYER_9_OUTPUT_CH,
		.kernel_k									=	LAYER_10_KERNEL_K,
		.output_cols								=	LAYER_10_OUTPUT_COLS,
		.output_rows								=	LAYER_10_OUTPUT_ROWS,
		.output_ch									=	LAYER_10_OUTPUT_CH,
		.output_image								= 	layer_10_output_image,
		.number_of_entries_per_output_row			=	layer_10_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_10_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_10_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_10_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_10_weight_array_0,
		.weight_array_1							    =	layer_10_weight_array_1,
		.weight_array_2							    =	layer_10_weight_array_2,
		.number_of_entries_bias_array				= 	layer_10_number_of_entries_bias_array,
		.bias_array									=   layer_10_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_10_pre_reg,
		.intra_reg								    =	&layer_10_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_11 = {
		.number_of_entries_per_row					=	layer_10_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_10_total_number_of_output_activations_entries,
		.input_activations							=	layer_10_output_image,
		.input_cols									=	LAYER_10_OUTPUT_COLS,
		.input_rows									=	LAYER_10_OUTPUT_ROWS,
		.input_ch									=	LAYER_10_OUTPUT_CH,
		.kernel_k									=	LAYER_11_KERNEL_K,
		.output_cols								=	LAYER_11_OUTPUT_COLS,
		.output_rows								=	LAYER_11_OUTPUT_ROWS,
		.output_ch									=	LAYER_11_OUTPUT_CH,
		.output_image								= 	layer_11_output_image,
		.number_of_entries_per_output_row			=	layer_11_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_11_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_11_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_11_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_11_weight_array_0,
		.weight_array_1							    =	layer_11_weight_array_1,
		.weight_array_2							    =	layer_11_weight_array_2,
		.number_of_entries_bias_array				= 	layer_11_number_of_entries_bias_array,
		.bias_array									=   layer_11_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_11_pre_reg,
		.intra_reg								    =	&layer_11_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_12 = {
		.number_of_entries_per_row					=	layer_11_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_11_total_number_of_output_activations_entries,
		.input_activations							=	layer_11_output_image,
		.input_cols									=	LAYER_11_OUTPUT_COLS,
		.input_rows									=	LAYER_11_OUTPUT_ROWS,
		.input_ch									=	LAYER_11_OUTPUT_CH,
		.kernel_k									=	LAYER_12_KERNEL_K,
		.output_cols								=	LAYER_12_OUTPUT_COLS,
		.output_rows								=	LAYER_12_OUTPUT_ROWS,
		.output_ch									=	LAYER_12_OUTPUT_CH,
		.output_image								= 	layer_12_output_image,
		.number_of_entries_per_output_row			=	layer_12_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_12_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_12_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_12_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_12_weight_array_0,
		.weight_array_1							    =	layer_12_weight_array_1,
		.weight_array_2							    =	layer_12_weight_array_2,
		.number_of_entries_bias_array				= 	layer_12_number_of_entries_bias_array,
		.bias_array									=   layer_12_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_12_pre_reg,
		.intra_reg								    =	&layer_12_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};
	layer_config_t layer_13 = {
		.number_of_entries_per_row					=	layer_12_number_of_entries_per_output_row,
		.total_number_of_input_activations_entries	=	layer_12_total_number_of_output_activations_entries,
		.input_activations							=	layer_12_output_image,
		.input_cols									=	LAYER_12_OUTPUT_COLS,
		.input_rows									=	LAYER_12_OUTPUT_ROWS,
		.input_ch									=	LAYER_12_OUTPUT_CH,
		.kernel_k									=	LAYER_13_KERNEL_K,
		.output_cols								=	LAYER_13_OUTPUT_COLS,
		.output_rows								=	LAYER_13_OUTPUT_ROWS,
		.output_ch									=	LAYER_13_OUTPUT_CH,
		.output_image								= 	layer_13_output_image,
		.number_of_entries_per_output_row			=	layer_13_number_of_entries_per_output_row,
		.total_number_of_output_activations_entries =	layer_13_total_number_of_output_activations_entries,
		.ground_truth_output_activations			=	layer_13_output_ground_truth,
		.number_of_entries_per_weight_array		    =	layer_13_number_of_entries_per_weight_array,
		.weight_array_0							    =	layer_13_weight_array_0,
		.weight_array_1							    =	layer_13_weight_array_1,
		.weight_array_2							    =	layer_13_weight_array_2,
		.number_of_entries_bias_array				= 	layer_13_number_of_entries_bias_array,
		.bias_array									=   layer_13_bias_array,
		.number_of_pre_reg						    =	number_of_pre_reg,
		.number_of_intra_reg					    = 	number_of_intra_reg,
		.pre_reg								    = 	&layer_13_pre_reg,
		.intra_reg								    =	&layer_13_intra_reg,
		.strided									=   0,
		.input_layer								=	0
	};

	NN_config_t NN = {
			.axi_cdma_instance = &axi_cdma_0_instance,
			.layer_0 = &layer_0,
			.layer_1 = &layer_1,
			.layer_2 = &layer_2,
			.layer_3 = &layer_3,
			.layer_4 = &layer_4,
			.layer_5 = &layer_5,
			.layer_6 = &layer_6,
			.layer_7 = &layer_7,
			.layer_8 = &layer_8,
			.layer_9 = &layer_9,
			.layer_10 = &layer_10,
			.layer_11 = &layer_11,
			.layer_12 = &layer_12,
			.layer_13 = &layer_13
	};

	// To count external memory transfers
	uint64_t *final_layer_number_of_entries_per_row = malloc(LAYER_13_OUTPUT_ROWS*sizeof(uint64_t));
	double number_bytes_transferred_external_memory = 0;
	double number_accesses_to_external_memory = 0;
	Xil_DCacheFlush();
//	Xil_DCacheDisable();

	// int execution_returns [3] 	= {0,0,0};
	XTime execution_start_time, execution_end_time;

	print("Running ...\n\r");
	XTime_GetTime(&execution_start_time);

	execute_layer(&axi_cdma_0_instance, &layer_0 , layer_1.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 0 \n\r");
//	Xil_DCacheFlush();
	execute_layer(&axi_cdma_0_instance, &layer_1 , layer_2.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 1 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_2 , layer_3.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 2 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_3 , layer_4.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 3 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_4 , layer_5.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 4 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_5 , layer_6.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 5 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_6 , layer_7.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 6 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_7 , layer_8.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 7 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_8 , layer_9.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 8 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_9 , layer_10.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 9 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_10, layer_11.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 10 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_11, layer_12.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 11 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_12, layer_13.number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 12 \n\r");
	execute_layer(&axi_cdma_0_instance, &layer_13, final_layer_number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
	print("layer 13 \n\r");
	XTime_GetTime(&execution_end_time);
//	Xil_DCacheEnable();
	Xil_DCacheFlush();



	if(check_output(layer_0_output_image, layer_0_output_ground_truth, layer_0_total_number_of_output_activations_entries)) {
		printf("Layer 0 Integrity check failed !\n\r");
	} else {
		printf("Layer 0 values are correct!\n\r");
	}
	if(check_output(layer_1_output_image, layer_1_output_ground_truth, layer_1_total_number_of_output_activations_entries)) {
		printf("Layer 1 Integrity check failed !\n\r");
	} else {
		printf("Layer 1 values are correct!\n\r");
	}
//
//	if(check_output(layer_2_output_image, layer_2_output_ground_truth, layer_2_total_number_of_output_activations_entries)) {
//		printf("Layer 2 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 2 values are correct!\n\r");
//	}
//
//	if(check_output(layer_3_output_image, layer_3_output_ground_truth, layer_3_total_number_of_output_activations_entries)) {
//		printf("Layer 3 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 3 values are correct!\n\r");
//	}
//
//	if(check_output(layer_4_output_image, layer_4_output_ground_truth, layer_4_total_number_of_output_activations_entries)) {
//		printf("Layer 4 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 4 values are correct!\n\r");
//	}
//
//	if(check_output(layer_5_output_image, layer_5_output_ground_truth, layer_5_total_number_of_output_activations_entries)) {
//		printf("Layer 5 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 5 values are correct!\n\r");
//	}
//
//	if(check_output(layer_6_output_image, layer_6_output_ground_truth, layer_6_total_number_of_output_activations_entries)) {
//		printf("Layer 6 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 6 values are correct!\n\r");
//	}
//
//	if(check_output(layer_7_output_image, layer_7_output_ground_truth, layer_7_total_number_of_output_activations_entries)) {
//		printf("Layer 7 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 7 values are correct!\n\r");
//	}
//
//	if(check_output(layer_8_output_image, layer_8_output_ground_truth, layer_8_total_number_of_output_activations_entries)) {
//		printf("Layer 8 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 8 values are correct!\n\r");
//	}
//
//	if(check_output(layer_9_output_image, layer_9_output_ground_truth, layer_9_total_number_of_output_activations_entries)) {
//		printf("Layer 9 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 9 values are correct!\n\r");
//	}
//
//	if(check_output(layer_10_output_image, layer_10_output_ground_truth, layer_10_total_number_of_output_activations_entries)) {
//		printf("Layer 10 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 10 values are correct!\n\r");
//	}
//
//	if(check_output(layer_11_output_image, layer_11_output_ground_truth, layer_11_total_number_of_output_activations_entries)) {
//		printf("Layer 11 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 11 values are correct!\n\r");
//	}
//	if(check_output(layer_12_output_image, layer_12_output_ground_truth, layer_12_total_number_of_output_activations_entries)) {
//		printf("Layer 12 Integrity check failed !\n\r");
//	} else {
//		printf("Layer 12 values are correct!\n\r");
//	}
//	Xil_DCacheFlush();
	if(check_output(layer_13_output_image, layer_13_output_ground_truth, layer_13_total_number_of_output_activations_entries)) {
		printf("Layer 13 Integrity check failed !\n\r");
	} else {
		printf("Layer 13 values are correct!\n\r");
	}

	print("Summary: \n\r");
	printf("Total run time = %.7fs\n\r", 1.0 * (execution_end_time - execution_start_time) / COUNTS_PER_SECOND);
	printf("Bytes transferred = %.1f \n\r", number_bytes_transferred_external_memory);
	printf("Number of transfer requests = %.1f \n\r", number_accesses_to_external_memory);

//
//	u32 xcoi, ycoi;
//	u8 gray_value;
//	s8 value;
//	s8 sign;
//	float gray_value_float;
//	u32 lineStart = ((1080/2) + (128/2))*DEMO_STRIDE;
//	u32 xstart = (1920/2) + (128/2);
//	u8 word_i = 0;
//	uint64_t word;
//	printf("word_i=%d, word=%llX \n\r", word_i, word);
//	int shift_index = 56;
//	u8 sm;
//	uint16_t count = 0;
//	xcoi = 0;
//	ycoi = 0;
//	u8 sm_flag = 1;
//	while(ycoi != 32)
//	{
//		printf("ycoi=%d \n\r", ycoi);
//		shift_index = 56;
//		word = layer_13.output_image[word_i];
//		word_i += 1;
//		printf("word_i=%d, word=%llX \n\r", word_i, word);
//		while(1)
//		{
//			if(sm_flag==1){
//				sm = word >> shift_index;
////				printf("sm=%X \n\r", sm);
//				sm_flag = 0;
//				shift_index -= 8;
//				if(shift_index==-8){
////					shift_index = 56;
////					word = layer_13.output_image[word_i];
////					word_i += 1;
////					printf("word_i=%d, word=%llX \n\r", word_i, word);
//					break;
//				}
//			}
//			else{
//				if(sm==0){
//					value   = 0;
//					sm_flag = 1;
//					printf("value=%d \n\r", value);
//					gray_value = 0;
////					destFrame[xcoi + lineStart] = gray_value;     //Red
////					destFrame[xcoi + lineStart + 1] = gray_value; //Blue
////					destFrame[xcoi + lineStart + 2] = gray_value; //Green
//					xcoi += 3;
//					if(xcoi == 32*3){
//						xcoi = 0;
//						lineStart += DEMO_STRIDE;
//						ycoi += 1;
//						print("\n");
//						print("------- \n\r");
//						break;
//					}
//				}
//				else{
//					value = word >> shift_index;
//					printf("value=%d \n\r", value);
//					sm_flag = 1;
//					shift_index -= 8;
//					gray_value_float = tanh(value);
//					sign = gray_value_float>=0? 1 : -1;
//					gray_value_float = gray_value_float + sign - gray_value_float;
//					gray_value_float = 0.5*(gray_value_float+1);
//					gray_value = gray_value_float*255;
//					if(gray_value==255){
//						count += 1;
//					}
////					destFrame[xcoi + lineStart] = gray_value;     //Red
////					destFrame[xcoi + lineStart + 1] = gray_value; //Blue
////					destFrame[xcoi + lineStart + 2] = gray_value; //Green
//					xcoi += 3;
//					if(xcoi == 32*3){
//						xcoi = 0;
//						lineStart += DEMO_STRIDE;
//						ycoi += 1;
//						print("\n");
//						print("------- \n\r");
//						break;
//					}
//					if(shift_index==-8){
////						shift_index = 56;
////						word = layer_13.output_image[word_i];
////						word_i += 1;
////						printf("word_i=%d, word=%llX \n\r", word_i, word);
//						break;
//					}
//				}
//			}
//		}
//	}
//	printf("count=%d \n\r", count);



//	free(layer_0_output_image);
//	free(layer_1_output_image);
//	free(layer_2_output_image);
//	free(layer_3_output_image);
//	free(layer_4_output_image);
//	free(layer_5_output_image);
//	free(layer_6_output_image);
//	free(layer_7_output_image);
//	free(layer_8_output_image);
//	free(layer_9_output_image);
//	free(layer_10_output_image);
//	free(layer_11_output_image);

//	Xil_DCacheFlush();
//	Xil_DCacheEnable();

//	Xil_DCacheDisable();
//	return 0;

	DemoInitialize();
	run_letters_demo(&NN);
	// DemoRun(&NN);

	cleanup_platform();
	return 0;
}


void DemoInitialize()
{
	int Status;
	XAxiVdma_Config *vdmaConfig;
	int i;
	Xil_DCacheDisable();
	/*
	 * Initialize an array of pointers to the 3 frame buffers
	 */
	for (i = 0; i < DISPLAY_NUM_FRAMES; i++)
	{
		pFrames[i] = frameBuf[i];
	}

	/*
	 * Initialize a timer used for a simple delay
	 */
	TimerInitialize(SCU_TIMER_ID);

	/*
	 * Initialize VDMA driver
	 */
	vdmaConfig = XAxiVdma_LookupConfig(VDMA_ID);
	if (!vdmaConfig)
	{
		xil_printf("No video DMA found for ID %d\r\n", VDMA_ID);
		return;
	}
	Status = XAxiVdma_CfgInitialize(&vdma, vdmaConfig, vdmaConfig->BaseAddress);
	if (Status != XST_SUCCESS)
	{
		xil_printf("VDMA Configuration Initialization failed %d\r\n", Status);
		return;
	}

	/*
	 * Initialize the Display controller and start it
	 */
	Status = DisplayInitialize(&dispCtrl, &vdma, HDMI_OUT_VTC_ID, DYNCLK_BASEADDR, pFrames, DEMO_STRIDE);
	if (Status != XST_SUCCESS)
	{
		xil_printf("Display Ctrl initialization failed during demo initialization%d\r\n", Status);
		return;
	}
	Status = DisplayStart(&dispCtrl);
	if (Status != XST_SUCCESS)
	{
		xil_printf("Couldn't start display during demo initialization%d\r\n", Status);
		return;
	}

	/*
	 * Initialize the Interrupt controller and start it.
	 */
	if(interrupts_enabled==0){
		Status = fnInitInterruptController(&intc);
		if(Status != XST_SUCCESS) {
			xil_printf("Error initializing interrupts");
			return;
		}
		fnEnableInterrupts(&intc, &ivt[0], sizeof(ivt)/sizeof(ivt[0]));
	}


	/*
	 * Initialize the Video Capture device
	 */
	Status = VideoInitialize(&videoCapt, &intc, &vdma, HDMI_IN_GPIO_ID, HDMI_IN_VTC_ID, HDMI_IN_VTC_IRPT_ID, pFrames, DEMO_STRIDE, DEMO_START_ON_DET);
	if (Status != XST_SUCCESS)
	{
		xil_printf("Video Ctrl initialization failed during demo initialization%d\r\n", Status);
		return;
	}

	/*
	 * Set the Video Detect callback to trigger the menu to reset, displaying the new detected resolution
	 */
	VideoSetCallback(&videoCapt, DemoISR, &fRefresh);







//	Xil_DCacheFlush();
//	printf("state= %d", videoCapt.state);

	DemoPrintTest(dispCtrl.framePtr[dispCtrl.curFrame], dispCtrl.vMode.width, dispCtrl.vMode.height, dispCtrl.stride, DEMO_PATTERN_1);

//	printf("state= %d", videoCapt.state);

	return;
}

void run_letters_demo(NN_config_t *NN)
{
	int nextFrame = 0;
	char userInput = 0;

	Xil_DCacheEnable();
	DemoChangeRes_1080();

	DisplayChangeFrame(&dispCtrl, 2);
	u32 width = videoCapt.timing.HActiveVideo;
	u32 height = videoCapt.timing.VActiveVideo;

	u32 start_pixel_width = (width/LAYER_0_INPUT_COLS) / 2;
	u32 start_pixel_height = (height/LAYER_0_INPUT_ROWS) / 2;
	// run_NN(NN, test_image[6],  compressed_test_image[6]  , pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE, start_pixel_width, start_pixel_height);

	uint64_t debug_compressed_img[7000];
	compressed_grayscale_input_image(test_image[0], debug_compressed_img, 128, 128, 0, 0, 0);
	
	while(1){
		for (int i = 0; i < 100; ++i) { // run 100 images
			xil_printf("---------------- %d -----------------\r\n", i);
			run_NN(NN, test_image[i],  compressed_test_image[i]  , pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE, start_pixel_width, start_pixel_height);
			// imec_test_configuation_set_pattern(i%3);
		}
	}
}

void DemoRun(NN_config_t *NN)
{
	int nextFrame = 0;
	char userInput = 0;

	Xil_DCacheEnable();

	/* Flush UART FIFO */
	while (XUartPs_IsReceiveData(UART_BASEADDR))
	{
		XUartPs_ReadReg(UART_BASEADDR, XUARTPS_FIFO_OFFSET);
	}

	while (userInput != 'q')
	{
		fRefresh = 0;
		DemoPrintMenu();

		/* Wait for data on UART */
		while (!XUartPs_IsReceiveData(UART_BASEADDR) && !fRefresh)
		{}
//		{xil_printf("waiting for input... \n");}

		/* Store the first character in the UART receive FIFO and echo it */
		if (XUartPs_IsReceiveData(UART_BASEADDR))
		{
			userInput = XUartPs_ReadReg(UART_BASEADDR, XUARTPS_FIFO_OFFSET);
			xil_printf("%c \n", userInput);
		}
		else  //Refresh triggered by video detect interrupt
		{
			userInput = 'r';
		}
//		Xil_DCacheInvalidateRange((unsigned int) 0x00118380, DEMO_MAX_FRAME);

		printf("User input is: %c \r\n", userInput);
		printf("state= %d \r\n", videoCapt.state);

		switch (userInput)
		{
		case '1':
			DemoChangeRes();
			break;
		case '2':
			nextFrame = dispCtrl.curFrame + 1;
			if (nextFrame >= DISPLAY_NUM_FRAMES)
			{
				nextFrame = 1;
			}
			DisplayChangeFrame(&dispCtrl, nextFrame);
			break;
		case '3':
			xil_printf("inside case 3 \r\n");
			DemoPrintTest(pFrames[dispCtrl.curFrame], dispCtrl.vMode.width, dispCtrl.vMode.height, DEMO_STRIDE, DEMO_PATTERN_0);
			break;
		case '4':
			xil_printf("inside case 4 \r\n");
			DemoPrintTest(pFrames[dispCtrl.curFrame], dispCtrl.vMode.width, dispCtrl.vMode.height, DEMO_STRIDE, DEMO_PATTERN_1);
			break;
		case '5':
			if (videoCapt.state == VIDEO_STREAMING)
				VideoStop(&videoCapt);
			else
				VideoStart(&videoCapt);
			break;
		case '6':
			nextFrame = videoCapt.curFrame + 1;
			if (nextFrame >= DISPLAY_NUM_FRAMES)
			{
				nextFrame = 1;
			}
			VideoChangeFrame(&videoCapt, nextFrame);
			break;
		case '7':
			// nextFrame = DemoGetInactiveFrame(&dispCtrl, &videoCapt);
			// VideoStop(&videoCapt);
			// DemoInvertFrame(pFrames[videoCapt.curFrame], pFrames[nextFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE);
			// VideoStart(&videoCapt);
			// DisplayChangeFrame(&dispCtrl, nextFrame);


			// nextFrame = DemoGetInactiveFrame(&dispCtrl, &videoCapt);
			VideoChangeFrame(&videoCapt, 1);
			DisplayChangeFrame(&dispCtrl, 3);
			// sleep(2);
			// VideoChangeFrame(&videoCapt, 1);
			// DisplayChangeFrame(&dispCtrl, 3);
			// sleep(2);
			while(1){
				// DemoScaleFrame(pFrames[videoCapt.curFrame], pFrames[2], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, 128, 128, DEMO_STRIDE);
				// DemoScaleFrame(pFrames[dispCtrl.curFrame], pFrames[2], 64, 64, 7, 14, DEMO_STRIDE);
				// to_Edges(pFrames[2], pFrames[dispCtrl.curFrame], 7, 14, DEMO_STRIDE);
				// DemoScaleFrame(pFrames[2], pFrames[dispCtrl.curFrame], 64, 64, 7, 14, DEMO_STRIDE);
				// stimulate(pFrames[dispCtrl.curFrame], 7, 14, DEMO_STRIDE);

				run_camera_NN(NN, pFrames[videoCapt.curFrame], pFrames[dispCtrl.curFrame], 128, 128, DEMO_STRIDE, 0, 0);
				// run_NN(NN, pFrames[videoCapt.curFrame], pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE, start_pixel_width, start_pixel_height);

				// DemoInvertFrame(pFrames[videoCapt.curFrame], pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE);
				// to_Gray_Scale(pFrames[videoCapt.curFrame], pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE);
				// to_Gray_Scale(pFrames[dispCtrl.curFrame], pFrames[dispCtrl.curFrame], 128, 128, DEMO_STRIDE);
				// to_Edges(pFrames[videoCapt.curFrame], pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE);
				// to_Edges(pFrames[2], pFrames[2], 16, 16, DEMO_STRIDE);

			}

// 			VideoChangeFrame(&videoCapt, 1);
// 			DisplayChangeFrame(&dispCtrl, 2);
// 			u32 width = videoCapt.timing.HActiveVideo;
// 			u32 height = videoCapt.timing.VActiveVideo;

// 			u32 start_pixel_width = (width/LAYER_0_INPUT_COLS) / 2;
// 			u32 start_pixel_height = (height/LAYER_0_INPUT_ROWS) / 2;
// //			break;
// 			while(1){
// //				run_NN(NN, pFrames[videoCapt.curFrame], pFrames[dispCtrl.curFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE, start_pixel_width, start_pixel_height);

// 			}
// //			VideoStop(&videoCapt);
// //			DemoInvertFrame(pFrames[videoCapt.curFrame], pFrames[nextFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, DEMO_STRIDE);
// //			VideoStart(&videoCapt);
// //			DisplayChangeFrame(&dispCtrl, nextFrame);


			break;
		case '8':
			nextFrame = DemoGetInactiveFrame(&dispCtrl, &videoCapt);
//			nextFrame = videoCapt.curFrame + 1;
//			if (nextFrame >= DISPLAY_NUM_FRAMES)
//			{
//				nextFrame = 1;
//			}
			VideoStop(&videoCapt);
			DemoScaleFrame(pFrames[videoCapt.curFrame], pFrames[nextFrame], videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo, dispCtrl.vMode.width, dispCtrl.vMode.height, DEMO_STRIDE);
			VideoStart(&videoCapt);
			DisplayChangeFrame(&dispCtrl, nextFrame);
			break;
		case 'q':
			break;
		case 'r':
			break;
		default :
			xil_printf("\n\rInvalid Selection");
			TimerDelay(500000);
		}

		xil_printf("After case statement \r\n");
	}

	return;
}

void DemoPrintMenu()
{
//	xil_printf("\x1B[H"); //Set cursor to top left of terminal
//	xil_printf("\x1B[2J"); //Clear terminal
	xil_printf("**************************************************\n\r");
	xil_printf("*                ZYBO Video Demo                 *\n\r");
	xil_printf("**************************************************\n\r");
	xil_printf("*Display Resolution: %28s*\n\r", dispCtrl.vMode.label);
	printf("*Display Pixel Clock Freq. (MHz): %15.3f*\n\r", dispCtrl.pxlFreq);
	xil_printf("*Display Frame Index: %27d*\n\r", dispCtrl.curFrame);
	if (videoCapt.state == VIDEO_DISCONNECTED) xil_printf("*Video Capture Resolution: %22s*\n\r", "!HDMI UNPLUGGED!");
	else xil_printf("*Video Capture Resolution: %17dx%-4d*\n\r", videoCapt.timing.HActiveVideo, videoCapt.timing.VActiveVideo);
	xil_printf("*Video Frame Index: %29d*\n\r", videoCapt.curFrame);
	xil_printf("**************************************************\n\r");
	xil_printf("\n\r");
	xil_printf("1 - Change Display Resolution\n\r");
	xil_printf("2 - Change Display Framebuffer Index\n\r");
	xil_printf("3 - Print Blended Test Pattern to Display Framebuffer\n\r");
	xil_printf("4 - Print Color Bar Test Pattern to Display Framebuffer\n\r");
	xil_printf("5 - Start/Stop Video stream into Video Framebuffer\n\r");
	xil_printf("6 - Change Video Framebuffer Index\n\r");
	xil_printf("7 - Grab Video Frame and invert colors\n\r");
	xil_printf("8 - Grab Video Frame and scale to Display resolution\n\r");
	xil_printf("q - Quit\n\r");
	xil_printf("\n\r");
	xil_printf("\n\r");
	xil_printf("Enter a selection:");
}

void DemoChangeRes()
{
	int fResSet = 0;
	int status;
	char userInput = 0;

	/* Flush UART FIFO */
	while (XUartPs_IsReceiveData(UART_BASEADDR))
	{
		XUartPs_ReadReg(UART_BASEADDR, XUARTPS_FIFO_OFFSET);
	}

	while (!fResSet)
	{
		DemoCRMenu();

		/* Wait for data on UART */
		while (!XUartPs_IsReceiveData(UART_BASEADDR))
		{}

		/* Store the first character in the UART recieve FIFO and echo it */
		userInput = XUartPs_ReadReg(UART_BASEADDR, XUARTPS_FIFO_OFFSET);
		xil_printf("%c", userInput);
		status = XST_SUCCESS;
		switch (userInput)
		{
		case '1':
			status = DisplayStop(&dispCtrl);
			DisplaySetMode(&dispCtrl, &VMODE_640x480);
			DisplayStart(&dispCtrl);
			fResSet = 1;
			break;
		case '2':
			status = DisplayStop(&dispCtrl);
			DisplaySetMode(&dispCtrl, &VMODE_800x600);
			DisplayStart(&dispCtrl);
			fResSet = 1;
			break;
		case '3':
			status = DisplayStop(&dispCtrl);
			DisplaySetMode(&dispCtrl, &VMODE_1280x720);
			DisplayStart(&dispCtrl);
			fResSet = 1;
			break;
		case '4':
			status = DisplayStop(&dispCtrl);
			DisplaySetMode(&dispCtrl, &VMODE_1280x1024);
			DisplayStart(&dispCtrl);
			fResSet = 1;
			break;
		case '5':
			status = DisplayStop(&dispCtrl);
			DisplaySetMode(&dispCtrl, &VMODE_1600x900);
			DisplayStart(&dispCtrl);
			fResSet = 1;
			break;
		case '6':
			status = DisplayStop(&dispCtrl);
			DisplaySetMode(&dispCtrl, &VMODE_1920x1080);
			DisplayStart(&dispCtrl);
			fResSet = 1;
			break;
		case 'q':
			fResSet = 1;
			break;
		default :
			xil_printf("\n\rInvalid Selection");
			TimerDelay(500000);
		}
		if (status == XST_DMA_ERROR)
		{
			xil_printf("\n\rWARNING: AXI VDMA Error detected and cleared\n\r");
		}
	}
}


void DemoChangeRes_1080()
{
	int fResSet = 0;
	int status;
	char userInput = 0;

	status = DisplayStop(&dispCtrl);
	DisplaySetMode(&dispCtrl, &VMODE_1920x1080);
	DisplayStart(&dispCtrl);
	fResSet = 1;


}

void DemoCRMenu()
{
//	xil_printf("\x1B[H"); //Set cursor to top left of terminal
//	xil_printf("\x1B[2J"); //Clear terminal
	xil_printf("**************************************************\n\r");
	xil_printf("*                ZYBO Video Demo                 *\n\r");
	xil_printf("**************************************************\n\r");
	xil_printf("*Current Resolution: %28s*\n\r", dispCtrl.vMode.label);
	printf("*Pixel Clock Freq. (MHz): %23.3f*\n\r", dispCtrl.pxlFreq);
	xil_printf("**************************************************\n\r");
	xil_printf("\n\r");
	xil_printf("1 - %s\n\r", VMODE_640x480.label);
	xil_printf("2 - %s\n\r", VMODE_800x600.label);
	xil_printf("3 - %s\n\r", VMODE_1280x720.label);
	xil_printf("4 - %s\n\r", VMODE_1280x1024.label);
	xil_printf("5 - %s\n\r", VMODE_1600x900.label);
	xil_printf("6 - %s\n\r", VMODE_1920x1080.label);
	xil_printf("q - Quit (don't change resolution)\n\r");
	xil_printf("\n\r");
	xil_printf("Select a new resolution:");
}

int DemoGetInactiveFrame(DisplayCtrl *DispCtrlPtr, VideoCapture *VideoCaptPtr)
{
	int i;
	for (i=1; i<DISPLAY_NUM_FRAMES; i++)
	{
		if (DispCtrlPtr->curFrame == i && DispCtrlPtr->state == DISPLAY_RUNNING)
		{
			continue;
		}
		else if (VideoCaptPtr->curFrame == i && VideoCaptPtr->state == VIDEO_STREAMING)
		{
			continue;
		}
		else
		{
			return i;
		}
	}
	xil_printf("Unreachable error state reached. All buffers are in use.\r\n");
	return 0;
}

void DemoInvertFrame(u8 *srcFrame, u8 *destFrame, u32 width, u32 height, u32 stride)
{
	u32 xcoi, ycoi;
	u32 lineStart = 0;
//	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	for(ycoi = 0; ycoi < height; ycoi++)
	{
		for(xcoi = 0; xcoi < (width * 3); xcoi+=3)
		{
			destFrame[xcoi + lineStart] = ~srcFrame[xcoi + lineStart];         //Red
			destFrame[xcoi + lineStart + 1] = ~srcFrame[xcoi + lineStart + 1]; //Blue
			destFrame[xcoi + lineStart + 2] = ~srcFrame[xcoi + lineStart + 2]; //Green
		}
		lineStart += stride;
	}
	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
//	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
}


void to_Gray_Scale(u8 *srcFrame, u8 *destFrame, u32 width, u32 height, u32 stride, u32 start_pixel_width, u32 start_pixel_height)
{
	u32 xcoi, ycoi;

	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u8 gray_value;
//	for(ycoi = 0; ycoi < height; ycoi++)
//	{
//		for(xcoi = 0; xcoi < (width * 3); xcoi+=3)
//		{
//			gray_value =  (srcFrame[xcoi + lineStart] + srcFrame[xcoi + lineStart + 1] + srcFrame[xcoi + lineStart + 2])/3;
//			destFrame[xcoi + lineStart] = gray_value;         //Red
//			destFrame[xcoi + lineStart + 1] = gray_value; //Blue
//			destFrame[xcoi + lineStart + 2] = gray_value; //Green
//		}
//		lineStart += stride;
//	}
//	printf("width=%d \r\n", width);
//	printf("height=%d \r\n", height);
//	printf("start_pixel_width=%d \r\n", start_pixel_width);
//	printf("start_pixel_height=%d \r\n", start_pixel_height);
//	printf("ycoi=%d \r\n", start_pixel_height*LAYER_0_INPUT_ROWS);
//	printf("xcoi=%d \r\n", (start_pixel_width* LAYER_0_INPUT_COLS * 3));

	// u32 lineStart = ((1080/2)-(128/2))*stride;
	// u32 xstart = (1920/2) - (128/2);
	u32 lineStart = 0;
	u32 xstart = 0;
//	u32 lineStart = 0;
	for(ycoi = 0; ycoi < 128; ycoi++)
	{
		for(xcoi = xstart*3 ; xcoi < ((xstart+128) * 3); xcoi+=3)
		{
			gray_value =  (srcFrame[xcoi + lineStart] + srcFrame[xcoi + lineStart + 1] + srcFrame[xcoi + lineStart + 2])/3;
			destFrame[xcoi + lineStart] = gray_value;     //Red
			destFrame[xcoi + lineStart + 1] = gray_value; //Blue
			destFrame[xcoi + lineStart + 2] = gray_value; //Green
		}
		lineStart += stride;
	}
	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
}

/*
 * Bilinear interpolation algorithm. Assumes both frames have the same stride.
 */
void ScaleFrame(u8 *srcFrame, u8 *destFrame, u32 srcWidth, u32 srcHeight, u32 destWidth, u32 destHeight, u32 stride)
{
	float xInc, yInc; // Width/height of a destination frame pixel in the source frame coordinate system
	float xcoSrc, ycoSrc; // Location of the destination pixel being operated on in the source frame coordinate system
	float x1y1, x2y1, x1y2, x2y2; //Used to store the color data of the four nearest source pixels to the destination pixel
	int ix1y1, ix2y1, ix1y2, ix2y2; //indexes into the source frame for the four nearest source pixels to the destination pixel
	float xDist, yDist; //distances between destination pixel and x1y1 source pixels in source frame coordinate system

	int xcoDest, ycoDest; // Location of the destination pixel being operated on in the destination coordinate system
	int iy1; //Used to store the index of the first source pixel in the line with y1
	int iDest; //index of the pixel data in the destination frame being operated on

	int i;

	xInc = ((float) srcWidth - 1.0) / ((float) destWidth);
	yInc = ((float) srcHeight - 1.0) / ((float) destHeight);

	ycoSrc = 0.0;
	for (ycoDest = 0; ycoDest < destHeight; ycoDest++)
	{
		iy1 = ((int) ycoSrc) * stride;
		yDist = ycoSrc - ((float) ((int) ycoSrc));

		/*
		 * Save some cycles in the loop below by presetting the destination
		 * index to the first pixel in the current line
		 */
		iDest = ycoDest * stride;

		xcoSrc = 0.0;
		for (xcoDest = 0; xcoDest < destWidth; xcoDest++)
		{
			ix1y1 = iy1 + ((int) xcoSrc) * 3;
			ix2y1 = ix1y1 + 3;
			ix1y2 = ix1y1 + stride;
			ix2y2 = ix1y1 + stride + 3;

			xDist = xcoSrc - ((float) ((int) xcoSrc));

			/*
			 * For loop handles all three colors
			 */
			for (i = 0; i < 3; i++)
			{
				x1y1 = (float) srcFrame[ix1y1 + i];
				x2y1 = (float) srcFrame[ix2y1 + i];
				x1y2 = (float) srcFrame[ix1y2 + i];
				x2y2 = (float) srcFrame[ix2y2 + i];

				/*
				 * Bilinear interpolation function
				 */
				destFrame[iDest] = (u8) ((1.0-yDist)*((1.0-xDist)*x1y1+xDist*x2y1) + yDist*((1.0-xDist)*x1y2+xDist*x2y2));
				iDest++;
			}
			xcoSrc += xInc;
		}
		ycoSrc += yInc;
	}

	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);

	return;
}

void to_Edges(u8 *srcFrame, u8 *destFrame, u32 width, u32 height, u32 stride)
{
	u32 xcoi, ycoi;
	u8 height_ = height+2;
	u8 width_ = width+2;
	// u8 gray_frame[height_*width_] = {0}; // padding
	// u8 *gray_frame = malloc(height_*width_*sizeof(u8)); // padding 
	u8 *gray_frame = calloc(height_*width_, sizeof(u8)); // padding 
	u8 edge_frame[height][width];
	int x_index = 1;
	int y_index = 1;
	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u8 gray_value;
	u8 edge_value;
	// u32 lineStart = ((1080/2)-(128/2))*stride;
	// u32 xstart = (1920/2) - (128/2);
	u32 lineStart = 0;
	u32 xstart = 0;

	for(ycoi = 0; ycoi < height; ycoi++)
	{
		for(xcoi = 0 ; xcoi < width; xcoi+=1)
		{
			gray_value =  (srcFrame[3*(xcoi+xstart) + lineStart] + srcFrame[3*(xcoi+xstart) + lineStart + 1] + srcFrame[3*(xcoi+xstart) + lineStart + 2])/3;
			gray_frame[x_index + y_index*width_] = gray_value;
			x_index++;
		}
		x_index = 1;
		y_index++;
		lineStart += stride;
	}

	// xil_printf("width=%d, height=%d \r\n", width, height);


	// //// print phosphenes
	// for (size_t ycoi = 0; ycoi < 128; ycoi++)
	// {
	// 	for (size_t xcoi = 0; xcoi < 128; xcoi++)
	// 	{	
	// 		xil_printf("%2d ",gray_frame[xcoi + ycoi*128] );
	// 	}
	// 	xil_printf("\r\n");	
	// }	
	// sleep(3);


	// for(ycoi = 1; ycoi < 128-1; ycoi++)
	// {
	// 	for(xcoi = 1 ; xcoi < 128-1; xcoi+=1)
	// 	{
	// 		//  E(C) = abs ( A(N) - A(S) ) + abs ( A(E) - A(W) )
	// 		edge_value =  abs(gray_frame[xcoi + ycoi*128 - 128] - gray_frame[xcoi + ycoi*128 + 128]) + abs(gray_frame[xcoi + ycoi*128 + 1] - gray_frame[xcoi + ycoi*128 - 1]);
	// 		destFrame[3*(xcoi+xstart) + lineStart] 	  = edge_value; //Red
	// 		destFrame[3*(xcoi+xstart) + lineStart + 1] = edge_value; //Blue
	// 		destFrame[3*(xcoi+xstart) + lineStart + 2] = edge_value; //Green
	// 	}
	// 	lineStart += stride;
	// }

	x_index = 1;
	y_index = 1;
	lineStart = 0;
	for(ycoi = 0; ycoi < height; ycoi++)
	{	
		for(xcoi = 0 ; xcoi < (width * 3); xcoi+=3)
		{	
			// if(y_index==128 || x_index==128){
			// 	edge_value = 0;	
			// }
			// else {
			// 	edge_value =  abs(gray_frame[x_index + y_index*128 - 128] - gray_frame[x_index + y_index*128 + 128]) + abs(gray_frame[x_index + y_index*128 + 1] - gray_frame[x_index + y_index*128 - 1]);
			// }
			edge_value =  abs(gray_frame[x_index + y_index*width_ - width_] - gray_frame[x_index + y_index*width_ + width_]) + abs(gray_frame[x_index + y_index*width_ + 1] - gray_frame[x_index + y_index*width_ - 1]);
			edge_frame[y_index-1][x_index-1] = edge_value;
			x_index++;
			destFrame[xcoi + lineStart] = edge_value;     //Red
			destFrame[xcoi + lineStart + 1] = edge_value; //Blue
			destFrame[xcoi + lineStart + 2] = edge_value; //Green

			// xil_printf("2. x_index=%d, y_index=%d \r\n", x_index, y_index);
			// xil_printf("2. width=%d, height=%d \r\n", width, height);
			// xil_printf("2. xcoi=%d, ycoi=%d \r\n", xcoi, ycoi);
			// usleep(1000*200);

		}
		x_index = 1;
		y_index++;
		lineStart += stride;
	}
	// xil_printf("------- \r\n");

	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);

	

	// int y_offset = 0;
	// int x_offset = 0;
	// unsigned char os_cfg_reg[8][16] = {0};
	// //col 1
	// os_cfg_reg[0][15] = edge_frame[0 + y_offset][0 + x_offset];
	// os_cfg_reg[4][15] = edge_frame[1 + y_offset][0 + x_offset];
	// os_cfg_reg[0][8]  = edge_frame[2 + y_offset][0 + x_offset];
	// os_cfg_reg[4][8]  = edge_frame[3 + y_offset][0 + x_offset];
	// os_cfg_reg[0][1]  = edge_frame[4 + y_offset][0 + x_offset];
	// os_cfg_reg[4][1]  = edge_frame[5 + y_offset][0 + x_offset];
	// os_cfg_reg[1][10] = edge_frame[6 + y_offset][0 + x_offset];
	// os_cfg_reg[5][10] = edge_frame[7 + y_offset][0 + x_offset];
	// os_cfg_reg[1][3]  = edge_frame[8 + y_offset][0 + x_offset];
	// os_cfg_reg[5][3]  = edge_frame[9 + y_offset][0 + x_offset];
	// os_cfg_reg[2][10] = edge_frame[10 + y_offset][0 + x_offset];
	// os_cfg_reg[6][10] = edge_frame[11 + y_offset][0 + x_offset];
	// os_cfg_reg[3][11] = edge_frame[12 + y_offset][0 + x_offset];
	// os_cfg_reg[7][11] = edge_frame[13 + y_offset][0 + x_offset];

	// //col 2
	// os_cfg_reg[0][14] = edge_frame[0 + y_offset] [1 + x_offset];
	// os_cfg_reg[4][14] = edge_frame[1 + y_offset] [1 + x_offset];
	// os_cfg_reg[0][7]  = edge_frame[2 + y_offset] [1 + x_offset];
	// os_cfg_reg[4][7]  = edge_frame[3 + y_offset] [1 + x_offset];
	// os_cfg_reg[0][0]  = edge_frame[4 + y_offset] [1 + x_offset];
	// os_cfg_reg[4][0]  = edge_frame[5 + y_offset] [1 + x_offset];
	// os_cfg_reg[1][9]  = edge_frame[6 + y_offset] [1 + x_offset];
	// os_cfg_reg[5][9]  = edge_frame[7 + y_offset] [1 + x_offset];
	// os_cfg_reg[1][2]  = edge_frame[8 + y_offset] [1 + x_offset];
	// os_cfg_reg[5][2]  = edge_frame[9 + y_offset] [1 + x_offset];
	// os_cfg_reg[2][9]  = edge_frame[10 + y_offset][1 + x_offset];
	// os_cfg_reg[6][9]  = edge_frame[11 + y_offset][1 + x_offset];
	// os_cfg_reg[3][10] = edge_frame[12 + y_offset][1 + x_offset];
	// os_cfg_reg[7][10] = edge_frame[13 + y_offset][1 + x_offset];

	// //col 3
	// os_cfg_reg[0][13] = edge_frame[0 + y_offset] [2 + x_offset];
	// os_cfg_reg[4][13] = edge_frame[1 + y_offset] [2 + x_offset];
	// os_cfg_reg[0][6]  = edge_frame[2 + y_offset] [2 + x_offset];
	// os_cfg_reg[4][6]  = edge_frame[3 + y_offset] [2 + x_offset];
	// os_cfg_reg[1][15] = edge_frame[4 + y_offset] [2 + x_offset];
	// os_cfg_reg[5][15] = edge_frame[5 + y_offset] [2 + x_offset];
	// os_cfg_reg[1][8]  = edge_frame[6 + y_offset] [2 + x_offset];
	// os_cfg_reg[5][8]  = edge_frame[7 + y_offset] [2 + x_offset];
	// os_cfg_reg[1][1]  = edge_frame[8 + y_offset] [2 + x_offset];
	// os_cfg_reg[5][1]  = edge_frame[9 + y_offset] [2 + x_offset];
	// os_cfg_reg[2][8]  = edge_frame[10 + y_offset][2 + x_offset];
	// os_cfg_reg[6][8]  = edge_frame[11 + y_offset][2 + x_offset];
	// os_cfg_reg[3][9]  = edge_frame[12 + y_offset][2 + x_offset];
	// os_cfg_reg[7][9]  = edge_frame[13 + y_offset][2 + x_offset];

	// //col 4
	// os_cfg_reg[0][12] = edge_frame[0 + y_offset] [3 + x_offset];
	// os_cfg_reg[4][12] = edge_frame[1 + y_offset] [3 + x_offset];
	// os_cfg_reg[0][5]  = edge_frame[2 + y_offset] [3 + x_offset];
	// os_cfg_reg[4][5]  = edge_frame[3 + y_offset] [3 + x_offset];
	// os_cfg_reg[1][14] = edge_frame[4 + y_offset] [3 + x_offset];
	// os_cfg_reg[5][14] = edge_frame[5 + y_offset] [3 + x_offset];
	// os_cfg_reg[1][7]  = edge_frame[6 + y_offset] [3 + x_offset];
	// os_cfg_reg[5][7]  = edge_frame[7 + y_offset] [3 + x_offset];
	// os_cfg_reg[1][0]  = edge_frame[8 + y_offset] [3 + x_offset];
	// os_cfg_reg[5][0]  = edge_frame[9 + y_offset] [3 + x_offset];
	// os_cfg_reg[2][7]  = edge_frame[10 + y_offset][3 + x_offset];
	// os_cfg_reg[6][7]  = edge_frame[11 + y_offset][3 + x_offset];
	// os_cfg_reg[3][8]  = edge_frame[12 + y_offset][3 + x_offset];
	// os_cfg_reg[7][8]  = edge_frame[13 + y_offset][3 + x_offset];

	// //col 5
	// os_cfg_reg[0][11] = edge_frame[0 + y_offset] [4 + x_offset];
	// os_cfg_reg[4][11] = edge_frame[1 + y_offset] [4 + x_offset];
	// os_cfg_reg[0][4]  = edge_frame[2 + y_offset] [4 + x_offset];
	// os_cfg_reg[4][4]  = edge_frame[3 + y_offset] [4 + x_offset];
	// os_cfg_reg[1][13] = edge_frame[4 + y_offset] [4 + x_offset];
	// os_cfg_reg[5][13] = edge_frame[5 + y_offset] [4 + x_offset];
	// os_cfg_reg[1][6]  = edge_frame[6 + y_offset] [4 + x_offset];
	// os_cfg_reg[5][6]  = edge_frame[7 + y_offset] [4 + x_offset];
	// os_cfg_reg[2][13] = edge_frame[8 + y_offset] [4 + x_offset];
	// os_cfg_reg[6][13] = edge_frame[9 + y_offset] [4 + x_offset];
	// os_cfg_reg[2][6]  = edge_frame[10 + y_offset][4 + x_offset];
	// os_cfg_reg[6][6]  = edge_frame[11 + y_offset][4 + x_offset];
	// os_cfg_reg[3][7]  = edge_frame[12 + y_offset][4 + x_offset];
	// os_cfg_reg[7][7]  = edge_frame[13 + y_offset][4 + x_offset];

	// //col 6
	// os_cfg_reg[0][10] = edge_frame[0 + y_offset] [5 + x_offset];
	// os_cfg_reg[4][10] = edge_frame[1 + y_offset] [5 + x_offset];
	// os_cfg_reg[0][3]  = edge_frame[2 + y_offset] [5 + x_offset];
	// os_cfg_reg[4][3]  = edge_frame[3 + y_offset] [5 + x_offset];
	// os_cfg_reg[1][12] = edge_frame[4 + y_offset] [5 + x_offset];
	// os_cfg_reg[5][12] = edge_frame[5 + y_offset] [5 + x_offset];
	// os_cfg_reg[1][5]  = edge_frame[6 + y_offset] [5 + x_offset];
	// os_cfg_reg[5][5]  = edge_frame[7 + y_offset] [5 + x_offset];
	// os_cfg_reg[2][12] = edge_frame[8 + y_offset] [5 + x_offset];
	// os_cfg_reg[6][12] = edge_frame[9 + y_offset] [5 + x_offset];
	// os_cfg_reg[2][5]  = edge_frame[10 + y_offset][5 + x_offset];
	// os_cfg_reg[6][5]  = edge_frame[11 + y_offset][5 + x_offset];
	// os_cfg_reg[3][6]  = edge_frame[12 + y_offset][5 + x_offset];
	// os_cfg_reg[7][6]  = edge_frame[13 + y_offset][5 + x_offset];

	// //col 7
	// os_cfg_reg[0][9]  = edge_frame[0 + y_offset] [6 + x_offset];
	// os_cfg_reg[4][9]  = edge_frame[1 + y_offset] [6 + x_offset];
	// os_cfg_reg[0][2]  = edge_frame[2 + y_offset] [6 + x_offset];
	// os_cfg_reg[4][2]  = edge_frame[3 + y_offset] [6 + x_offset];
	// os_cfg_reg[1][11] = edge_frame[4 + y_offset] [6 + x_offset];
	// os_cfg_reg[5][11] = edge_frame[5 + y_offset] [6 + x_offset];
	// os_cfg_reg[1][4]  = edge_frame[6 + y_offset] [6 + x_offset];
	// os_cfg_reg[5][4]  = edge_frame[7 + y_offset] [6 + x_offset];
	// os_cfg_reg[2][11] = edge_frame[8 + y_offset] [6 + x_offset];
	// os_cfg_reg[6][11] = edge_frame[9 + y_offset] [6 + x_offset];
	// os_cfg_reg[2][4]  = edge_frame[10 + y_offset][6 + x_offset];
	// os_cfg_reg[6][4]  = edge_frame[11 + y_offset][6 + x_offset];
	// os_cfg_reg[3][5]  = edge_frame[12 + y_offset][6 + x_offset];
	// os_cfg_reg[7][5]  = edge_frame[13 + y_offset][6 + x_offset];

	// // for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+14; phosphene_y_index++)
	// // {
	// // 	for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+7; phosphene_x_index++)
	// // 	{	
	// // 		if(edge_frame[phosphene_y_index][phosphene_x_index])
	// // 			xil_printf("o");
	// // 		else
	// // 			xil_printf(".");
	// // 		// xil_printf("%3d ", edge_frame[phosphene_y_index][phosphene_x_index]? 111:0);
	// // 	}
	// // 	xil_printf("\r\n");	
	// // }
	// imec_phosphene_14x7_output(os_cfg_reg);
	free(gray_frame);
	

	
}

void stimulate(u8* srcFrame, u32 width, u32 height, u32 stride){

	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u32 xcoi, ycoi;
	u8 edge_frame[height][width];
	int x_index = 1;
	int y_index = 1;
	u32 lineStart = 0;
	u32 xstart = 0;
	for(ycoi = 0; ycoi < height; ycoi++)
	{
		for(xcoi = 0 ; xcoi < (width * 3); xcoi+=3)
		{
			// gray_value =  (srcFrame[3*(xcoi+xstart) + lineStart] + srcFrame[3*(xcoi+xstart) + lineStart + 1] + srcFrame[3*(xcoi+xstart) + lineStart + 2])/3;
			edge_frame[y_index][x_index] = srcFrame[xcoi + lineStart];
			x_index++;
		}
		x_index = 1;
		y_index++;
		lineStart += stride;
	}

	int y_offset = 0;
	int x_offset = 0;
	unsigned char os_cfg_reg[8][16] = {0};
	//col 1
	os_cfg_reg[0][15] = edge_frame[0 + y_offset][0 + x_offset];
	os_cfg_reg[4][15] = edge_frame[1 + y_offset][0 + x_offset];
	os_cfg_reg[0][8]  = edge_frame[2 + y_offset][0 + x_offset];
	os_cfg_reg[4][8]  = edge_frame[3 + y_offset][0 + x_offset];
	os_cfg_reg[0][1]  = edge_frame[4 + y_offset][0 + x_offset];
	os_cfg_reg[4][1]  = edge_frame[5 + y_offset][0 + x_offset];
	os_cfg_reg[1][10] = edge_frame[6 + y_offset][0 + x_offset];
	os_cfg_reg[5][10] = edge_frame[7 + y_offset][0 + x_offset];
	os_cfg_reg[1][3]  = edge_frame[8 + y_offset][0 + x_offset];
	os_cfg_reg[5][3]  = edge_frame[9 + y_offset][0 + x_offset];
	os_cfg_reg[2][10] = edge_frame[10 + y_offset][0 + x_offset];
	os_cfg_reg[6][10] = edge_frame[11 + y_offset][0 + x_offset];
	os_cfg_reg[3][11] = edge_frame[12 + y_offset][0 + x_offset];
	os_cfg_reg[7][11] = edge_frame[13 + y_offset][0 + x_offset];

	//col 2
	os_cfg_reg[0][14] = edge_frame[0 + y_offset] [1 + x_offset];
	os_cfg_reg[4][14] = edge_frame[1 + y_offset] [1 + x_offset];
	os_cfg_reg[0][7]  = edge_frame[2 + y_offset] [1 + x_offset];
	os_cfg_reg[4][7]  = edge_frame[3 + y_offset] [1 + x_offset];
	os_cfg_reg[0][0]  = edge_frame[4 + y_offset] [1 + x_offset];
	os_cfg_reg[4][0]  = edge_frame[5 + y_offset] [1 + x_offset];
	os_cfg_reg[1][9]  = edge_frame[6 + y_offset] [1 + x_offset];
	os_cfg_reg[5][9]  = edge_frame[7 + y_offset] [1 + x_offset];
	os_cfg_reg[1][2]  = edge_frame[8 + y_offset] [1 + x_offset];
	os_cfg_reg[5][2]  = edge_frame[9 + y_offset] [1 + x_offset];
	os_cfg_reg[2][9]  = edge_frame[10 + y_offset][1 + x_offset];
	os_cfg_reg[6][9]  = edge_frame[11 + y_offset][1 + x_offset];
	os_cfg_reg[3][10] = edge_frame[12 + y_offset][1 + x_offset];
	os_cfg_reg[7][10] = edge_frame[13 + y_offset][1 + x_offset];

	//col 3
	os_cfg_reg[0][13] = edge_frame[0 + y_offset] [2 + x_offset];
	os_cfg_reg[4][13] = edge_frame[1 + y_offset] [2 + x_offset];
	os_cfg_reg[0][6]  = edge_frame[2 + y_offset] [2 + x_offset];
	os_cfg_reg[4][6]  = edge_frame[3 + y_offset] [2 + x_offset];
	os_cfg_reg[1][15] = edge_frame[4 + y_offset] [2 + x_offset];
	os_cfg_reg[5][15] = edge_frame[5 + y_offset] [2 + x_offset];
	os_cfg_reg[1][8]  = edge_frame[6 + y_offset] [2 + x_offset];
	os_cfg_reg[5][8]  = edge_frame[7 + y_offset] [2 + x_offset];
	os_cfg_reg[1][1]  = edge_frame[8 + y_offset] [2 + x_offset];
	os_cfg_reg[5][1]  = edge_frame[9 + y_offset] [2 + x_offset];
	os_cfg_reg[2][8]  = edge_frame[10 + y_offset][2 + x_offset];
	os_cfg_reg[6][8]  = edge_frame[11 + y_offset][2 + x_offset];
	os_cfg_reg[3][9]  = edge_frame[12 + y_offset][2 + x_offset];
	os_cfg_reg[7][9]  = edge_frame[13 + y_offset][2 + x_offset];

	//col 4
	os_cfg_reg[0][12] = edge_frame[0 + y_offset] [3 + x_offset];
	os_cfg_reg[4][12] = edge_frame[1 + y_offset] [3 + x_offset];
	os_cfg_reg[0][5]  = edge_frame[2 + y_offset] [3 + x_offset];
	os_cfg_reg[4][5]  = edge_frame[3 + y_offset] [3 + x_offset];
	os_cfg_reg[1][14] = edge_frame[4 + y_offset] [3 + x_offset];
	os_cfg_reg[5][14] = edge_frame[5 + y_offset] [3 + x_offset];
	os_cfg_reg[1][7]  = edge_frame[6 + y_offset] [3 + x_offset];
	os_cfg_reg[5][7]  = edge_frame[7 + y_offset] [3 + x_offset];
	os_cfg_reg[1][0]  = edge_frame[8 + y_offset] [3 + x_offset];
	os_cfg_reg[5][0]  = edge_frame[9 + y_offset] [3 + x_offset];
	os_cfg_reg[2][7]  = edge_frame[10 + y_offset][3 + x_offset];
	os_cfg_reg[6][7]  = edge_frame[11 + y_offset][3 + x_offset];
	os_cfg_reg[3][8]  = edge_frame[12 + y_offset][3 + x_offset];
	os_cfg_reg[7][8]  = edge_frame[13 + y_offset][3 + x_offset];

	//col 5
	os_cfg_reg[0][11] = edge_frame[0 + y_offset] [4 + x_offset];
	os_cfg_reg[4][11] = edge_frame[1 + y_offset] [4 + x_offset];
	os_cfg_reg[0][4]  = edge_frame[2 + y_offset] [4 + x_offset];
	os_cfg_reg[4][4]  = edge_frame[3 + y_offset] [4 + x_offset];
	os_cfg_reg[1][13] = edge_frame[4 + y_offset] [4 + x_offset];
	os_cfg_reg[5][13] = edge_frame[5 + y_offset] [4 + x_offset];
	os_cfg_reg[1][6]  = edge_frame[6 + y_offset] [4 + x_offset];
	os_cfg_reg[5][6]  = edge_frame[7 + y_offset] [4 + x_offset];
	os_cfg_reg[2][13] = edge_frame[8 + y_offset] [4 + x_offset];
	os_cfg_reg[6][13] = edge_frame[9 + y_offset] [4 + x_offset];
	os_cfg_reg[2][6]  = edge_frame[10 + y_offset][4 + x_offset];
	os_cfg_reg[6][6]  = edge_frame[11 + y_offset][4 + x_offset];
	os_cfg_reg[3][7]  = edge_frame[12 + y_offset][4 + x_offset];
	os_cfg_reg[7][7]  = edge_frame[13 + y_offset][4 + x_offset];

	//col 6
	os_cfg_reg[0][10] = edge_frame[0 + y_offset] [5 + x_offset];
	os_cfg_reg[4][10] = edge_frame[1 + y_offset] [5 + x_offset];
	os_cfg_reg[0][3]  = edge_frame[2 + y_offset] [5 + x_offset];
	os_cfg_reg[4][3]  = edge_frame[3 + y_offset] [5 + x_offset];
	os_cfg_reg[1][12] = edge_frame[4 + y_offset] [5 + x_offset];
	os_cfg_reg[5][12] = edge_frame[5 + y_offset] [5 + x_offset];
	os_cfg_reg[1][5]  = edge_frame[6 + y_offset] [5 + x_offset];
	os_cfg_reg[5][5]  = edge_frame[7 + y_offset] [5 + x_offset];
	os_cfg_reg[2][12] = edge_frame[8 + y_offset] [5 + x_offset];
	os_cfg_reg[6][12] = edge_frame[9 + y_offset] [5 + x_offset];
	os_cfg_reg[2][5]  = edge_frame[10 + y_offset][5 + x_offset];
	os_cfg_reg[6][5]  = edge_frame[11 + y_offset][5 + x_offset];
	os_cfg_reg[3][6]  = edge_frame[12 + y_offset][5 + x_offset];
	os_cfg_reg[7][6]  = edge_frame[13 + y_offset][5 + x_offset];

	//col 7
	os_cfg_reg[0][9]  = edge_frame[0 + y_offset] [6 + x_offset];
	os_cfg_reg[4][9]  = edge_frame[1 + y_offset] [6 + x_offset];
	os_cfg_reg[0][2]  = edge_frame[2 + y_offset] [6 + x_offset];
	os_cfg_reg[4][2]  = edge_frame[3 + y_offset] [6 + x_offset];
	os_cfg_reg[1][11] = edge_frame[4 + y_offset] [6 + x_offset];
	os_cfg_reg[5][11] = edge_frame[5 + y_offset] [6 + x_offset];
	os_cfg_reg[1][4]  = edge_frame[6 + y_offset] [6 + x_offset];
	os_cfg_reg[5][4]  = edge_frame[7 + y_offset] [6 + x_offset];
	os_cfg_reg[2][11] = edge_frame[8 + y_offset] [6 + x_offset];
	os_cfg_reg[6][11] = edge_frame[9 + y_offset] [6 + x_offset];
	os_cfg_reg[2][4]  = edge_frame[10 + y_offset][6 + x_offset];
	os_cfg_reg[6][4]  = edge_frame[11 + y_offset][6 + x_offset];
	os_cfg_reg[3][5]  = edge_frame[12 + y_offset][6 + x_offset];
	os_cfg_reg[7][5]  = edge_frame[13 + y_offset][6 + x_offset];

	// for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+14; phosphene_y_index++)
	// {
	// 	for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+7; phosphene_x_index++)
	// 	{	
	// 		if(edge_frame[phosphene_y_index][phosphene_x_index])
	// 			xil_printf("o");
	// 		else
	// 			xil_printf(".");
	// 		// xil_printf("%3d ", edge_frame[phosphene_y_index][phosphene_x_index]? 111:0);
	// 	}
	// 	xil_printf("\r\n");	
	// }
	imec_phosphene_14x7_output(os_cfg_reg);
}

void compressed_grayscale_input_image(u8 *srcFrame, uint64_t *destFrame, u32 width, u32 height, u32 stride, u32 start_pixel_width, u32 start_pixel_height)
{
	u32 xcoi, ycoi;

//	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u8 gray_value;
	u32 lineStart = ((1080/2)-(128/2))*stride;
	u32 xstart = (1920/2) - (128/2);
	// u32 lineStart = 0;
	// u32 xstart = 0;
	u32 dest_line_start = 0;
	u32 dest_i = 0;
	u8 compressed_word_remaining_bytes = AXI_BIT_WIDTH/8;
	u8 sm;
	int shift_index = 56;
	uint64_t val = 0;
	destFrame[dest_i] = 0;

	for(ycoi = 0; ycoi < height; ycoi++)
	{
		for(xcoi = 0 ; xcoi < width; xcoi++)
		{
			// gray_value =  (srcFrame[xcoi + lineStart] + srcFrame[xcoi + lineStart + 1] + srcFrame[xcoi + lineStart + 2])/3;
			gray_value =  srcFrame[xcoi + ycoi*width];
			if(gray_value==0){
				sm = 0;
				val = (uint64_t)sm << shift_index;
//				destFrame[dest_i] += val;
				destFrame[dest_i] |= val;
				shift_index -= 8;
				// check
				if(shift_index==-8){
					shift_index = 56;
					dest_i++;
					destFrame[dest_i] = 0;
				}
			}
			else {
				sm = 0x80;
				val = (uint64_t)sm << shift_index;
//				destFrame[dest_i] += val;
				destFrame[dest_i] |= val;
				shift_index -= 8;
				// check
				if(shift_index==-8){
					shift_index = 56;
					dest_i++;
					destFrame[dest_i] = 0;
				}
				val = (uint64_t)gray_value << shift_index;
//				destFrame[dest_i] += val;
				destFrame[dest_i] |= val;
				shift_index -= 8;
				// check
				if(shift_index==-8){
					shift_index = 56;
					dest_i++;
					destFrame[dest_i] = 0;
				}
			}
		}
		lineStart += stride;
	}
	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
//	return dest_i;
}


void compressed_input_image(u8 *srcFrame, uint64_t *destFrame, u32 width, u32 height, u32 stride, u32 start_pixel_width, u32 start_pixel_height)
{
	u32 xcoi, ycoi;

//	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u8 gray_value;
	u32 lineStart = ((1080/2)-(128/2))*stride;
	u32 xstart = (1920/2) - (128/2);
	// u32 lineStart = 0;
	// u32 xstart = 0;
	u32 dest_line_start = 0;
	u32 dest_i = 0;
	u8 compressed_word_remaining_bytes = AXI_BIT_WIDTH/8;
	u8 sm;
	int shift_index = 56;
	uint64_t val = 0;
	destFrame[dest_i] = 0;

	for(ycoi = 0; ycoi < height; ycoi++)
	{
		for(xcoi = xstart*3 ; xcoi < ((xstart+width) * 3); xcoi+=3)
		{
			gray_value =  (srcFrame[xcoi + lineStart] + srcFrame[xcoi + lineStart + 1] + srcFrame[xcoi + lineStart + 2])/3;
			if(gray_value==0){
				sm = 0;
				val = (uint64_t)sm << shift_index;
//				destFrame[dest_i] += val;
				destFrame[dest_i] |= val;
				shift_index -= 8;
				// check
				if(shift_index==-8){
					shift_index = 56;
					dest_i++;
					destFrame[dest_i] = 0;
				}
			}
			else {
				sm = 0x80;
				val = (uint64_t)sm << shift_index;
//				destFrame[dest_i] += val;
				destFrame[dest_i] |= val;
				shift_index -= 8;
				// check
				if(shift_index==-8){
					shift_index = 56;
					dest_i++;
					destFrame[dest_i] = 0;
				}
				val = (uint64_t)gray_value << shift_index;
//				destFrame[dest_i] += val;
				destFrame[dest_i] |= val;
				shift_index -= 8;
				// check
				if(shift_index==-8){
					shift_index = 56;
					dest_i++;
					destFrame[dest_i] = 0;
				}
			}
		}
		lineStart += stride;
	}
	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
//	return dest_i;
}

/*
 * Bilinear interpolation algorithm. Assumes both frames have the same stride.
 */
void DemoScaleFrame(u8 *srcFrame, u8 *destFrame, u32 srcWidth, u32 srcHeight, u32 destWidth, u32 destHeight, u32 stride)
{
	float xInc, yInc; // Width/height of a destination frame pixel in the source frame coordinate system
	float xcoSrc, ycoSrc; // Location of the destination pixel being operated on in the source frame coordinate system
	float x1y1, x2y1, x1y2, x2y2; //Used to store the color data of the four nearest source pixels to the destination pixel
	int ix1y1, ix2y1, ix1y2, ix2y2; //indexes into the source frame for the four nearest source pixels to the destination pixel
	float xDist, yDist; //distances between destination pixel and x1y1 source pixels in source frame coordinate system

	int xcoDest, ycoDest; // Location of the destination pixel being operated on in the destination coordinate system
	int iy1; //Used to store the index of the first source pixel in the line with y1
	int iDest; //index of the pixel data in the destination frame being operated on

	int i;

	xInc = ((float) srcWidth - 1.0) / ((float) destWidth);
	yInc = ((float) srcHeight - 1.0) / ((float) destHeight);

	ycoSrc = 0.0;
	for (ycoDest = 0; ycoDest < destHeight; ycoDest++)
	{
		iy1 = ((int) ycoSrc) * stride;
		yDist = ycoSrc - ((float) ((int) ycoSrc));

		/*
		 * Save some cycles in the loop below by presetting the destination
		 * index to the first pixel in the current line
		 */
		iDest = ycoDest * stride;

		xcoSrc = 0.0;
		for (xcoDest = 0; xcoDest < destWidth; xcoDest++)
		{
			ix1y1 = iy1 + ((int) xcoSrc) * 3;
			ix2y1 = ix1y1 + 3;
			ix1y2 = ix1y1 + stride;
			ix2y2 = ix1y1 + stride + 3;

			xDist = xcoSrc - ((float) ((int) xcoSrc));

			/*
			 * For loop handles all three colors
			 */
			for (i = 0; i < 3; i++)
			{
				x1y1 = (float) srcFrame[ix1y1 + i];
				x2y1 = (float) srcFrame[ix2y1 + i];
				x1y2 = (float) srcFrame[ix1y2 + i];
				x2y2 = (float) srcFrame[ix2y2 + i];

				/*
				 * Bilinear interpolation function
				 */
				destFrame[iDest] = (u8) ((1.0-yDist)*((1.0-xDist)*x1y1+xDist*x2y1) + yDist*((1.0-xDist)*x1y2+xDist*x2y2));
				iDest++;
			}
			xcoSrc += xInc;
		}
		ycoSrc += yInc;
	}

	/*
	 * Flush the framebuffer memory range to ensure changes are written to the
	 * actual memory, and therefore accessible by the VDMA.
	 */
	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);

	return;
}

void DemoPrintTest(u8 *frame, u32 width, u32 height, u32 stride, int pattern)
{
	u32 xcoi, ycoi;
	u32 iPixelAddr;
	u8 wRed, wBlue, wGreen;
	u32 wCurrentInt;
	double fRed, fBlue, fGreen, fColor;
	u32 xLeft, xMid, xRight, xInt;
	u32 yMid, yInt;
	double xInc, yInc;


	switch (pattern)
	{
	case DEMO_PATTERN_0:

		xInt = width / 4; //Four intervals, each with width/4 pixels
		xLeft = xInt * 3;
		xMid = xInt * 2 * 3;
		xRight = xInt * 3 * 3;
		xInc = 256.0 / ((double) xInt); //256 color intensities are cycled through per interval (overflow must be caught when color=256.0)

		yInt = height / 2; //Two intervals, each with width/2 lines
		yMid = yInt;
		yInc = 256.0 / ((double) yInt); //256 color intensities are cycled through per interval (overflow must be caught when color=256.0)

		fBlue = 0.0;
		fRed = 256.0;
		for(xcoi = 0; xcoi < (width*3); xcoi+=3)
		{
			/*
			 * Convert color intensities to integers < 256, and trim values >=256
			 */
			wRed = (fRed >= 256.0) ? 255 : ((u8) fRed);
			wBlue = (fBlue >= 256.0) ? 255 : ((u8) fBlue);
			iPixelAddr = xcoi;
			fGreen = 0.0;
			for(ycoi = 0; ycoi < height; ycoi++)
			{

				wGreen = (fGreen >= 256.0) ? 255 : ((u8) fGreen);
				frame[iPixelAddr] = wRed;
				frame[iPixelAddr + 1] = wBlue;
				frame[iPixelAddr + 2] = wGreen;
				if (ycoi < yMid)
				{
					fGreen += yInc;
				}
				else
				{
					fGreen -= yInc;
				}

				/*
				 * This pattern is printed one vertical line at a time, so the address must be incremented
				 * by the stride instead of just 1.
				 */
				iPixelAddr += stride;
			}

			if (xcoi < xLeft)
			{
				fBlue = 0.0;
				fRed -= xInc;
			}
			else if (xcoi < xMid)
			{
				fBlue += xInc;
				fRed += xInc;
			}
			else if (xcoi < xRight)
			{
				fBlue -= xInc;
				fRed -= xInc;
			}
			else
			{
				fBlue += xInc;
				fRed = 0;
			}
		}
		/*
		 * Flush the framebuffer memory range to ensure changes are written to the
		 * actual memory, and therefore accessible by the VDMA.
		 */
		Xil_DCacheFlushRange((unsigned int) frame, DEMO_MAX_FRAME);
		break;
	case DEMO_PATTERN_1:

		xInt = width / 7; //Seven intervals, each with width/7 pixels
		xInc = 256.0 / ((double) xInt); //256 color intensities per interval. Notice that overflow is handled for this pattern.

		fColor = 0.0;
		wCurrentInt = 1;
		for(xcoi = 0; xcoi < (width*3); xcoi+=3)
		{

			/*
			 * Just draw white in the last partial interval (when width is not divisible by 7)
			 */
			if (wCurrentInt > 7)
			{
				wRed = 255;
				wBlue = 255;
				wGreen = 255;
			}
			else
			{
				if (wCurrentInt & 0b001)
					wRed = (u8) fColor;
				else
					wRed = 0;

				if (wCurrentInt & 0b010)
					wBlue = (u8) fColor;
				else
					wBlue = 0;

				if (wCurrentInt & 0b100)
					wGreen = (u8) fColor;
				else
					wGreen = 0;
			}

			iPixelAddr = xcoi;

			for(ycoi = 0; ycoi < height; ycoi++)
			{
				frame[iPixelAddr] = wRed;
				frame[iPixelAddr + 1] = wBlue;
				frame[iPixelAddr + 2] = wGreen;
				/*
				 * This pattern is printed one vertical line at a time, so the address must be incremented
				 * by the stride instead of just 1.
				 */
				iPixelAddr += stride;
			}

			fColor += xInc;
			if (fColor >= 256.0)
			{
				fColor = 0.0;
				wCurrentInt++;
			}
		}
		/*
		 * Flush the framebuffer memory range to ensure changes are written to the
		 * actual memory, and therefore accessible by the VDMA.
		 */
		Xil_DCacheFlushRange((unsigned int) frame, DEMO_MAX_FRAME);
		break;
	default :
		xil_printf("Error: invalid pattern passed to DemoPrintTest");
	}
}

void DemoISR(void *callBackRef, void *pVideo)
{
	char *data = (char *) callBackRef;
	*data = 1; //set fRefresh to 1
}


void run_NN(NN_config_t *NN, u8 *srcFrame, uint64_t *compressed_src_frame, u8 *destFrame, u32 width, u32 height, u32 stride, u32 start_pixel_width, u32 start_pixel_height)
{
	XTime execution_start_time, execution_end_time;
	double number_bytes_transferred_external_memory = 0;
	double number_accesses_to_external_memory = 0;
	uint64_t *final_layer_number_of_entries_per_row = malloc(LAYER_13_OUTPUT_ROWS*sizeof(uint64_t));
//	print("Running ...\n\r");

	u32 xcoi, ycoi;
//	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u8 gray_value;
	u32 lineStart = ((1080/2)-(128/2))*stride;
	u32 xstart = (1920/2) - (128/2);
	u32 image_index = 0;
	for(ycoi = 0; ycoi < 128; ycoi++)
	{
		for(xcoi = xstart*3 ; xcoi < ((xstart+128) * 3); xcoi+=3)
		{
//			gray_value =  (srcFrame[xcoi + lineStart] + srcFrame[xcoi + lineStart + 1] + srcFrame[xcoi + lineStart + 2])/3;
			gray_value = srcFrame[image_index];
			image_index += 1;
			destFrame[xcoi + lineStart] = gray_value;     //Red
			destFrame[xcoi + lineStart + 1] = gray_value; //Blue
			destFrame[xcoi + lineStart + 2] = gray_value; //Green
		}
		lineStart += stride;
	}
	Xil_DCacheFlush();
//	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
//	to_Gray_Scale(srcFrame, destFrame, width, height, stride, start_pixel_width, start_pixel_height);

	XTime_GetTime(&execution_start_time);

	// This function is incorrect. TODO:: fix me
//	compressed_input_image(srcFrame, NN->layer_0->input_activations, width, height, stride, start_pixel_width, start_pixel_height);
	NN->layer_0->input_activations = compressed_src_frame;
	Xil_DCacheFlush();

	execute_layer(NN->axi_cdma_instance, NN->layer_0 , NN->layer_1->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 0 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_1 , NN->layer_2->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 1 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_2 , NN->layer_3->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 2 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_3 , NN->layer_4->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 3 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_4 , NN->layer_5->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 4 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_5 , NN->layer_6->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 5 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_6 , NN->layer_7->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);\
//	print("layer 6 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_7 , NN->layer_8->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 7 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_8 , NN->layer_9->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 8 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_9 , NN->layer_10->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 9 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_10, NN->layer_11->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 10 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_11, NN->layer_12->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 11 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_12, NN->layer_13->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 12 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_13, final_layer_number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 13 \n\r");


	XTime_GetTime(&execution_end_time);
	print("Summary: \n\r");
	printf("Total run time = %.7fs\n\r", 1.0 * (execution_end_time - execution_start_time) / COUNTS_PER_SECOND);
	printf("Bytes transferred = %.1f \n\r", number_bytes_transferred_external_memory);
	printf("Number of transfer requests = %.1f \n\r", number_accesses_to_external_memory);

	Xil_DCacheFlush();

	s8 value;
	s8 sign;
	float gray_value_float;
	lineStart = ((1080/2) + (128/2))*DEMO_STRIDE;
	xstart = (1920/2) + (128/2);
	u8 word_i = 0;
	uint64_t word;
//	printf("word_i=%d, word=%llX \n\r", word_i, word);
	int shift_index = 56;
	u8 sm;
	uint16_t count = 0;
	xcoi = xstart*3;
	ycoi = 0;
	u8 sm_flag = 1;
	u8 phosphenes [32][32] = {0};
	u8 phosphene_x_index = 0;
	u8 phosphene_y_index = 0;
	while(ycoi != 32)
	{
//		printf("ycoi=%d \n\r", ycoi);
		shift_index = 56;
		word = NN->layer_13->output_image[word_i];
		word_i += 1;
//		printf("word_i=%d, word=%llX \n\r", word_i, word);
		while(1)
		{
			if(sm_flag==1){
				sm = word >> shift_index;
//				printf("sm=%X \n\r", sm);
				sm_flag = 0;
				shift_index -= 8;
				if(shift_index==-8){
					break;
				}
			}
			else{
				if(sm==0){
					value   = 0;
					sm_flag = 1;
//					printf("value=%d \n\r", value);
					gray_value = 0;
					destFrame[xcoi + lineStart] = gray_value;     //Red
					destFrame[xcoi + lineStart + 1] = gray_value; //Blue
					destFrame[xcoi + lineStart + 2] = gray_value; //Green
					xcoi += 3;
					phosphene_x_index += 1;
					if(xcoi == (xstart+32)*3){
						xcoi = xstart*3;
						lineStart += DEMO_STRIDE;
						ycoi += 1;
						phosphene_x_index = 0;
						phosphene_y_index += 1;
//						print("\n");
//						print("------- \n\r");
						break;
					}
				}
				else{
					value = word >> shift_index;
//					printf("value=%d \n\r", value);
					sm_flag = 1;
					shift_index -= 8;
					gray_value_float = tanh(value);
					sign = gray_value_float>=0? 1 : -1;
					gray_value_float = gray_value_float + sign - gray_value_float;
					gray_value_float = 0.5*(gray_value_float+1);
					gray_value = gray_value_float*255;
					if(gray_value==255){
						count += 1;
					}
					destFrame[xcoi + lineStart] = gray_value;     //Red
					destFrame[xcoi + lineStart + 1] = gray_value; //Blue
					destFrame[xcoi + lineStart + 2] = gray_value; //Green
					phosphenes[phosphene_y_index][phosphene_x_index] = (int)gray_value_float;
					xcoi += 3;
					phosphene_x_index += 1;
					if(xcoi == (xstart+32)*3){
						xcoi = xstart*3;
						lineStart += DEMO_STRIDE;
						ycoi += 1;
						phosphene_x_index = 0;
						phosphene_y_index += 1;
//						print("\n");
//						print("------- \n\r");
						break;
					}
					if(shift_index==-8){
						break;
					}
				}
			}
		}
	}
//	printf("count=%d \n\r", count);
//	Xil_DCacheFlushRange((unsigned int) outputFrame, DEMO_MAX_FRAME);

//	destFrame = outputFrame;
	// Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
	Xil_DCacheFlush();

	//// print phosphenes
	for (size_t phosphene_y_index = 0; phosphene_y_index < 32; phosphene_y_index++)
	{
		for (size_t phosphene_x_index = 0; phosphene_x_index < 32; phosphene_x_index++)
		{	
			if(phosphenes[phosphene_y_index][phosphene_x_index])
				xil_printf("o");
			else
				xil_printf(".");
			// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
		}
		xil_printf("\r\n");	
	}	

	u8 downsized_phosphenes [16][16] = {0};
	for (size_t phosphene_y_index = 0; phosphene_y_index < 16; phosphene_y_index++)
	{
		for (size_t phosphene_x_index = 0; phosphene_x_index < 16; phosphene_x_index++)
		{	
			if((phosphenes[2*phosphene_y_index][2*phosphene_x_index] + phosphenes[2*phosphene_y_index][2*phosphene_x_index+1] + phosphenes[2*phosphene_y_index+1][2*phosphene_x_index] + phosphenes[2*phosphene_y_index+1][2*phosphene_x_index+1]) >= 1){
				downsized_phosphenes[phosphene_y_index][phosphene_x_index] = 1;
			}
			else {
				downsized_phosphenes[phosphene_y_index][phosphene_x_index] = 0;
			}
		}
	}	
	//// print phosphenes
	for (size_t phosphene_y_index = 0; phosphene_y_index < 16; phosphene_y_index++)
	{
		for (size_t phosphene_x_index = 0; phosphene_x_index < 16; phosphene_x_index++)
		{	
			if(downsized_phosphenes[phosphene_y_index][phosphene_x_index]>=1)
				xil_printf("o");
			else
				xil_printf(".");
			// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
		}
		xil_printf("\r\n");	
	}	
	
	// //// map phosphenes to LEDs
	// //// 7x5 LED display
	// //// offsets are used to select the region to crop and show on the LEDs
	// int y_offset = 32/2 - 7/2;
	// int x_offset = 32/2 - 5/2; 
	// unsigned char os_cfg_reg[3][16] = {0};
	// os_cfg_reg[0][15] = phosphenes[0 + y_offset][0 + x_offset];
	// os_cfg_reg[0][14] = phosphenes[1 + y_offset][0 + x_offset];
	// os_cfg_reg[0][13] = phosphenes[2 + y_offset][0 + x_offset];
	// os_cfg_reg[0][12] = phosphenes[3 + y_offset][0 + x_offset];
	// os_cfg_reg[0][11] = phosphenes[4 + y_offset][0 + x_offset];
	// os_cfg_reg[0][10] = phosphenes[5 + y_offset][0 + x_offset];
	// os_cfg_reg[0][9]  = phosphenes[6 + y_offset][0 + x_offset];

	// os_cfg_reg[0][7] = phosphenes[0 + y_offset][1 + x_offset];
	// os_cfg_reg[0][6] = phosphenes[1 + y_offset][1 + x_offset];
	// os_cfg_reg[0][5] = phosphenes[2 + y_offset][1 + x_offset];
	// os_cfg_reg[0][4] = phosphenes[3 + y_offset][1 + x_offset];
	// os_cfg_reg[0][3] = phosphenes[4 + y_offset][1 + x_offset];
	// os_cfg_reg[0][2] = phosphenes[5 + y_offset][1 + x_offset];
	// os_cfg_reg[0][1] = phosphenes[6 + y_offset][1 + x_offset];

	// os_cfg_reg[1][15] = phosphenes[0 + y_offset][2 + x_offset];
	// os_cfg_reg[1][14] = phosphenes[1 + y_offset][2 + x_offset];
	// os_cfg_reg[1][13] = phosphenes[2 + y_offset][2 + x_offset];
	// os_cfg_reg[1][12] = phosphenes[3 + y_offset][2 + x_offset];
	// os_cfg_reg[1][11] = phosphenes[4 + y_offset][2 + x_offset];
	// os_cfg_reg[1][10] = phosphenes[5 + y_offset][2 + x_offset];
	// os_cfg_reg[1][9]  = phosphenes[6 + y_offset][2 + x_offset];

	// os_cfg_reg[1][7] = phosphenes[0 + y_offset][3 + x_offset];
	// os_cfg_reg[1][6] = phosphenes[1 + y_offset][3 + x_offset];
	// os_cfg_reg[1][5] = phosphenes[2 + y_offset][3 + x_offset];
	// os_cfg_reg[1][4] = phosphenes[3 + y_offset][3 + x_offset];
	// os_cfg_reg[1][3] = phosphenes[4 + y_offset][3 + x_offset];
	// os_cfg_reg[1][2] = phosphenes[5 + y_offset][3 + x_offset];
	// os_cfg_reg[1][1] = phosphenes[6 + y_offset][3 + x_offset];

	// os_cfg_reg[2][13] = phosphenes[0 + y_offset][4 + x_offset];
	// os_cfg_reg[2][12] = phosphenes[1 + y_offset][4 + x_offset];
	// os_cfg_reg[2][11] = phosphenes[2 + y_offset][4 + x_offset];
	// os_cfg_reg[2][10] = phosphenes[3 + y_offset][4 + x_offset];
	// os_cfg_reg[2][9]  = phosphenes[4 + y_offset][4 + x_offset];
	// os_cfg_reg[2][8]  = phosphenes[5 + y_offset][4 + x_offset];
	// os_cfg_reg[2][7]  = phosphenes[6 + y_offset][4 + x_offset];
	// for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+7; phosphene_y_index++)
	// {
	// 	for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+5; phosphene_x_index++)
	// 	{	
	// 		if(phosphenes[phosphene_y_index][phosphene_x_index])
	// 			xil_printf("o");
	// 		else
	// 			xil_printf(".");
	// 		// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
	// 	}
	// 	xil_printf("\r\n");	
	// }
	// imec_phosphene_7x5_output(os_cfg_reg);

	//// map phosphenes to LEDs
	//// 14x7 LED display
	//// offsets are used to select the region to crop and show on the LEDs
	int y_offset = 1;
	int x_offset = 4;
	unsigned char os_cfg_reg[8][16] = {0};
	//col 1
	os_cfg_reg[0][15] = downsized_phosphenes[0 + y_offset][0 + x_offset];
	os_cfg_reg[4][15] = downsized_phosphenes[1 + y_offset][0 + x_offset];
	os_cfg_reg[0][8]  = downsized_phosphenes[2 + y_offset][0 + x_offset];
	os_cfg_reg[4][8]  = downsized_phosphenes[3 + y_offset][0 + x_offset];
	os_cfg_reg[0][1]  = downsized_phosphenes[4 + y_offset][0 + x_offset];
	os_cfg_reg[4][1]  = downsized_phosphenes[5 + y_offset][0 + x_offset];
	os_cfg_reg[1][10] = downsized_phosphenes[6 + y_offset][0 + x_offset];
	os_cfg_reg[5][10] = downsized_phosphenes[7 + y_offset][0 + x_offset];
	os_cfg_reg[1][3]  = downsized_phosphenes[8 + y_offset][0 + x_offset];
	os_cfg_reg[5][3]  = downsized_phosphenes[9 + y_offset][0 + x_offset];
	os_cfg_reg[2][10] = downsized_phosphenes[10 + y_offset][0 + x_offset];
	os_cfg_reg[6][10] = downsized_phosphenes[11 + y_offset][0 + x_offset];
	os_cfg_reg[3][11] = downsized_phosphenes[12 + y_offset][0 + x_offset];
	os_cfg_reg[7][11] = downsized_phosphenes[13 + y_offset][0 + x_offset];

	//col 2
	os_cfg_reg[0][14] = downsized_phosphenes[0 + y_offset] [1 + x_offset];
	os_cfg_reg[4][14] = downsized_phosphenes[1 + y_offset] [1 + x_offset];
	os_cfg_reg[0][7]  = downsized_phosphenes[2 + y_offset] [1 + x_offset];
	os_cfg_reg[4][7]  = downsized_phosphenes[3 + y_offset] [1 + x_offset];
	os_cfg_reg[0][0]  = downsized_phosphenes[4 + y_offset] [1 + x_offset];
	os_cfg_reg[4][0]  = downsized_phosphenes[5 + y_offset] [1 + x_offset];
	os_cfg_reg[1][9]  = downsized_phosphenes[6 + y_offset] [1 + x_offset];
	os_cfg_reg[5][9]  = downsized_phosphenes[7 + y_offset] [1 + x_offset];
	os_cfg_reg[1][2]  = downsized_phosphenes[8 + y_offset] [1 + x_offset];
	os_cfg_reg[5][2]  = downsized_phosphenes[9 + y_offset] [1 + x_offset];
	os_cfg_reg[2][9]  = downsized_phosphenes[10 + y_offset][1 + x_offset];
	os_cfg_reg[6][9]  = downsized_phosphenes[11 + y_offset][1 + x_offset];
	os_cfg_reg[3][10] = downsized_phosphenes[12 + y_offset][1 + x_offset];
	os_cfg_reg[7][10] = downsized_phosphenes[13 + y_offset][1 + x_offset];

	//col 3
	os_cfg_reg[0][13] = downsized_phosphenes[0 + y_offset] [2 + x_offset];
	os_cfg_reg[4][13] = downsized_phosphenes[1 + y_offset] [2 + x_offset];
	os_cfg_reg[0][6]  = downsized_phosphenes[2 + y_offset] [2 + x_offset];
	os_cfg_reg[4][6]  = downsized_phosphenes[3 + y_offset] [2 + x_offset];
	os_cfg_reg[1][15] = downsized_phosphenes[4 + y_offset] [2 + x_offset];
	os_cfg_reg[5][15] = downsized_phosphenes[5 + y_offset] [2 + x_offset];
	os_cfg_reg[1][8]  = downsized_phosphenes[6 + y_offset] [2 + x_offset];
	os_cfg_reg[5][8]  = downsized_phosphenes[7 + y_offset] [2 + x_offset];
	os_cfg_reg[1][1]  = downsized_phosphenes[8 + y_offset] [2 + x_offset];
	os_cfg_reg[5][1]  = downsized_phosphenes[9 + y_offset] [2 + x_offset];
	os_cfg_reg[2][8]  = downsized_phosphenes[10 + y_offset][2 + x_offset];
	os_cfg_reg[6][8]  = downsized_phosphenes[11 + y_offset][2 + x_offset];
	os_cfg_reg[3][9]  = downsized_phosphenes[12 + y_offset][2 + x_offset];
	os_cfg_reg[7][9]  = downsized_phosphenes[13 + y_offset][2 + x_offset];

	//col 4
	os_cfg_reg[0][12] = downsized_phosphenes[0 + y_offset] [3 + x_offset];
	os_cfg_reg[4][12] = downsized_phosphenes[1 + y_offset] [3 + x_offset];
	os_cfg_reg[0][5]  = downsized_phosphenes[2 + y_offset] [3 + x_offset];
	os_cfg_reg[4][5]  = downsized_phosphenes[3 + y_offset] [3 + x_offset];
	os_cfg_reg[1][14] = downsized_phosphenes[4 + y_offset] [3 + x_offset];
	os_cfg_reg[5][14] = downsized_phosphenes[5 + y_offset] [3 + x_offset];
	os_cfg_reg[1][7]  = downsized_phosphenes[6 + y_offset] [3 + x_offset];
	os_cfg_reg[5][7]  = downsized_phosphenes[7 + y_offset] [3 + x_offset];
	os_cfg_reg[1][0]  = downsized_phosphenes[8 + y_offset] [3 + x_offset];
	os_cfg_reg[5][0]  = downsized_phosphenes[9 + y_offset] [3 + x_offset];
	os_cfg_reg[2][7]  = downsized_phosphenes[10 + y_offset][3 + x_offset];
	os_cfg_reg[6][7]  = downsized_phosphenes[11 + y_offset][3 + x_offset];
	os_cfg_reg[3][8]  = downsized_phosphenes[12 + y_offset][3 + x_offset];
	os_cfg_reg[7][8]  = downsized_phosphenes[13 + y_offset][3 + x_offset];

	//col 5
	os_cfg_reg[0][11] = downsized_phosphenes[0 + y_offset] [4 + x_offset];
	os_cfg_reg[4][11] = downsized_phosphenes[1 + y_offset] [4 + x_offset];
	os_cfg_reg[0][4]  = downsized_phosphenes[2 + y_offset] [4 + x_offset];
	os_cfg_reg[4][4]  = downsized_phosphenes[3 + y_offset] [4 + x_offset];
	os_cfg_reg[1][13] = downsized_phosphenes[4 + y_offset] [4 + x_offset];
	os_cfg_reg[5][13] = downsized_phosphenes[5 + y_offset] [4 + x_offset];
	os_cfg_reg[1][6]  = downsized_phosphenes[6 + y_offset] [4 + x_offset];
	os_cfg_reg[5][6]  = downsized_phosphenes[7 + y_offset] [4 + x_offset];
	os_cfg_reg[2][13] = downsized_phosphenes[8 + y_offset] [4 + x_offset];
	os_cfg_reg[6][13] = downsized_phosphenes[9 + y_offset] [4 + x_offset];
	os_cfg_reg[2][6]  = downsized_phosphenes[10 + y_offset][4 + x_offset];
	os_cfg_reg[6][6]  = downsized_phosphenes[11 + y_offset][4 + x_offset];
	os_cfg_reg[3][7]  = downsized_phosphenes[12 + y_offset][4 + x_offset];
	os_cfg_reg[7][7]  = downsized_phosphenes[13 + y_offset][4 + x_offset];

	//col 6
	os_cfg_reg[0][10] = downsized_phosphenes[0 + y_offset] [5 + x_offset];
	os_cfg_reg[4][10] = downsized_phosphenes[1 + y_offset] [5 + x_offset];
	os_cfg_reg[0][3]  = downsized_phosphenes[2 + y_offset] [5 + x_offset];
	os_cfg_reg[4][3]  = downsized_phosphenes[3 + y_offset] [5 + x_offset];
	os_cfg_reg[1][12] = downsized_phosphenes[4 + y_offset] [5 + x_offset];
	os_cfg_reg[5][12] = downsized_phosphenes[5 + y_offset] [5 + x_offset];
	os_cfg_reg[1][5]  = downsized_phosphenes[6 + y_offset] [5 + x_offset];
	os_cfg_reg[5][5]  = downsized_phosphenes[7 + y_offset] [5 + x_offset];
	os_cfg_reg[2][12] = downsized_phosphenes[8 + y_offset] [5 + x_offset];
	os_cfg_reg[6][12] = downsized_phosphenes[9 + y_offset] [5 + x_offset];
	os_cfg_reg[2][5]  = downsized_phosphenes[10 + y_offset][5 + x_offset];
	os_cfg_reg[6][5]  = downsized_phosphenes[11 + y_offset][5 + x_offset];
	os_cfg_reg[3][6]  = downsized_phosphenes[12 + y_offset][5 + x_offset];
	os_cfg_reg[7][6]  = downsized_phosphenes[13 + y_offset][5 + x_offset];

	//col 7
	os_cfg_reg[0][9]  = downsized_phosphenes[0 + y_offset] [6 + x_offset];
	os_cfg_reg[4][9]  = downsized_phosphenes[1 + y_offset] [6 + x_offset];
	os_cfg_reg[0][2]  = downsized_phosphenes[2 + y_offset] [6 + x_offset];
	os_cfg_reg[4][2]  = downsized_phosphenes[3 + y_offset] [6 + x_offset];
	os_cfg_reg[1][11] = downsized_phosphenes[4 + y_offset] [6 + x_offset];
	os_cfg_reg[5][11] = downsized_phosphenes[5 + y_offset] [6 + x_offset];
	os_cfg_reg[1][4]  = downsized_phosphenes[6 + y_offset] [6 + x_offset];
	os_cfg_reg[5][4]  = downsized_phosphenes[7 + y_offset] [6 + x_offset];
	os_cfg_reg[2][11] = downsized_phosphenes[8 + y_offset] [6 + x_offset];
	os_cfg_reg[6][11] = downsized_phosphenes[9 + y_offset] [6 + x_offset];
	os_cfg_reg[2][4]  = downsized_phosphenes[10 + y_offset][6 + x_offset];
	os_cfg_reg[6][4]  = downsized_phosphenes[11 + y_offset][6 + x_offset];
	os_cfg_reg[3][5]  = downsized_phosphenes[12 + y_offset][6 + x_offset];
	os_cfg_reg[7][5]  = downsized_phosphenes[13 + y_offset][6 + x_offset];



	for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+14; phosphene_y_index++)
	{
		for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+7; phosphene_x_index++)
		{	
			if(downsized_phosphenes[phosphene_y_index][phosphene_x_index])
				xil_printf("o");
			else
				xil_printf(".");
			// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
		}
		xil_printf("\r\n");	
	}
	imec_phosphene_14x7_output(os_cfg_reg);

	//// map phosphenes to LEDs
	//// 7x14 LED display
	//// offsets are used to select the region to crop and show on the LEDs
	// int y_offset = 0;
	// int x_offset = 0;
	// unsigned char os_cfg_reg[8][16] = {0};
	// //col 1
	// os_cfg_reg[0][15] = phosphenes[0 + y_offset][0 + x_offset];
	// os_cfg_reg[0][14] = phosphenes[1 + y_offset][0 + x_offset];
	// os_cfg_reg[0][13] = phosphenes[2 + y_offset][0 + x_offset];
	// os_cfg_reg[0][12] = phosphenes[3 + y_offset][0 + x_offset];
	// os_cfg_reg[0][11] = phosphenes[4 + y_offset][0 + x_offset];
	// os_cfg_reg[0][10] = phosphenes[5 + y_offset][0 + x_offset];
	// os_cfg_reg[0][9]  = phosphenes[6 + y_offset][0 + x_offset];

	// //col 3
	// os_cfg_reg[0][8] = phosphenes[0 + y_offset][2 + x_offset];
	// os_cfg_reg[0][7] = phosphenes[1 + y_offset][2 + x_offset];
	// os_cfg_reg[0][6] = phosphenes[2 + y_offset][2 + x_offset];
	// os_cfg_reg[0][5] = phosphenes[3 + y_offset][2 + x_offset];
	// os_cfg_reg[0][4] = phosphenes[4 + y_offset][2 + x_offset];
	// os_cfg_reg[0][3] = phosphenes[5 + y_offset][2 + x_offset];
	// os_cfg_reg[0][2] = phosphenes[6 + y_offset][2 + x_offset];

	// //col 5
	// os_cfg_reg[0][1]  = phosphenes[0 + y_offset][4 + x_offset];
	// os_cfg_reg[0][0]  = phosphenes[1 + y_offset][4 + x_offset];
	// os_cfg_reg[1][15] = phosphenes[2 + y_offset][4 + x_offset];
	// os_cfg_reg[1][14] = phosphenes[3 + y_offset][4 + x_offset];
	// os_cfg_reg[1][13] = phosphenes[4 + y_offset][4 + x_offset];
	// os_cfg_reg[1][12] = phosphenes[5 + y_offset][4 + x_offset];
	// os_cfg_reg[1][11] = phosphenes[6 + y_offset][4 + x_offset];

	// //col 7
	// os_cfg_reg[1][10] = phosphenes[0 + y_offset][7 + x_offset];
	// os_cfg_reg[1][9]  = phosphenes[1 + y_offset][7 + x_offset];
	// os_cfg_reg[1][8]  = phosphenes[2 + y_offset][7 + x_offset];
	// os_cfg_reg[1][7]  = phosphenes[3 + y_offset][7 + x_offset];
	// os_cfg_reg[1][6]  = phosphenes[4 + y_offset][7 + x_offset];
	// os_cfg_reg[1][5]  = phosphenes[5 + y_offset][7 + x_offset];
	// os_cfg_reg[1][4]  = phosphenes[6 + y_offset][7 + x_offset];

	// //col 9
	// os_cfg_reg[1][3]  = phosphenes[0 + y_offset][8 + x_offset];
	// os_cfg_reg[1][2]  = phosphenes[1 + y_offset][8 + x_offset];
	// os_cfg_reg[1][1]  = phosphenes[2 + y_offset][8 + x_offset];
	// os_cfg_reg[1][0]  = phosphenes[3 + y_offset][8 + x_offset];
	// os_cfg_reg[2][13] = phosphenes[4 + y_offset][8 + x_offset];
	// os_cfg_reg[2][12] = phosphenes[5 + y_offset][8 + x_offset];
	// os_cfg_reg[2][11] = phosphenes[6 + y_offset][8 + x_offset];

	// //col 11
	// os_cfg_reg[2][10] = phosphenes[0 + y_offset][10 + x_offset];
	// os_cfg_reg[2][9]  = phosphenes[1 + y_offset][10 + x_offset];
	// os_cfg_reg[2][8]  = phosphenes[2 + y_offset][10 + x_offset];
	// os_cfg_reg[2][7]  = phosphenes[3 + y_offset][10 + x_offset];
	// os_cfg_reg[2][6]  = phosphenes[4 + y_offset][10 + x_offset];
	// os_cfg_reg[2][5]  = phosphenes[5 + y_offset][10 + x_offset];
	// os_cfg_reg[2][4]  = phosphenes[6 + y_offset][10 + x_offset];

	// //col 13
	// os_cfg_reg[3][11] = phosphenes[0 + y_offset][12 + x_offset];
	// os_cfg_reg[3][10] = phosphenes[1 + y_offset][12 + x_offset];
	// os_cfg_reg[3][9]  = phosphenes[2 + y_offset][12 + x_offset];
	// os_cfg_reg[3][8]  = phosphenes[3 + y_offset][12 + x_offset];
	// os_cfg_reg[3][7]  = phosphenes[4 + y_offset][12 + x_offset];
	// os_cfg_reg[3][6]  = phosphenes[5 + y_offset][12 + x_offset];
	// os_cfg_reg[3][5]  = phosphenes[6 + y_offset][12 + x_offset];

	// //col 2
	// os_cfg_reg[4][15] = phosphenes[0 + y_offset][1 + x_offset];
	// os_cfg_reg[4][14] = phosphenes[1 + y_offset][1 + x_offset];
	// os_cfg_reg[4][13] = phosphenes[2 + y_offset][1 + x_offset];
	// os_cfg_reg[4][12] = phosphenes[3 + y_offset][1 + x_offset];
	// os_cfg_reg[4][11] = phosphenes[4 + y_offset][1 + x_offset];
	// os_cfg_reg[4][10] = phosphenes[5 + y_offset][1 + x_offset];
	// os_cfg_reg[4][9]  = phosphenes[6 + y_offset][1 + x_offset];

	// //col 4
	// os_cfg_reg[4][8] = phosphenes[0 + y_offset][3 + x_offset];
	// os_cfg_reg[4][7] = phosphenes[1 + y_offset][3 + x_offset];
	// os_cfg_reg[4][6] = phosphenes[2 + y_offset][3 + x_offset];
	// os_cfg_reg[4][5] = phosphenes[3 + y_offset][3 + x_offset];
	// os_cfg_reg[4][4] = phosphenes[4 + y_offset][3 + x_offset];
	// os_cfg_reg[4][3] = phosphenes[5 + y_offset][3 + x_offset];
	// os_cfg_reg[4][2] = phosphenes[6 + y_offset][3 + x_offset];

	// //col 6
	// os_cfg_reg[4][1]  = phosphenes[0 + y_offset][5 + x_offset];
	// os_cfg_reg[4][0]  = phosphenes[1 + y_offset][5 + x_offset];
	// os_cfg_reg[5][15] = phosphenes[2 + y_offset][5 + x_offset];
	// os_cfg_reg[5][14] = phosphenes[3 + y_offset][5 + x_offset];
	// os_cfg_reg[5][13] = phosphenes[4 + y_offset][5 + x_offset];
	// os_cfg_reg[5][12] = phosphenes[5 + y_offset][5 + x_offset];
	// os_cfg_reg[5][11] = phosphenes[6 + y_offset][5 + x_offset];

	// //col 8
	// os_cfg_reg[5][10] = phosphenes[0 + y_offset][7 + x_offset];
	// os_cfg_reg[5][9]  = phosphenes[1 + y_offset][7 + x_offset];
	// os_cfg_reg[5][8]  = phosphenes[2 + y_offset][7 + x_offset];
	// os_cfg_reg[5][7]  = phosphenes[3 + y_offset][7 + x_offset];
	// os_cfg_reg[5][6]  = phosphenes[4 + y_offset][7 + x_offset];
	// os_cfg_reg[5][5]  = phosphenes[5 + y_offset][7 + x_offset];
	// os_cfg_reg[5][4]  = phosphenes[6 + y_offset][7 + x_offset];

	// //col 10
	// os_cfg_reg[5][3]  = phosphenes[0 + y_offset][9 + x_offset];
	// os_cfg_reg[5][2]  = phosphenes[1 + y_offset][9 + x_offset];
	// os_cfg_reg[5][1]  = phosphenes[2 + y_offset][9 + x_offset];
	// os_cfg_reg[5][0]  = phosphenes[3 + y_offset][9 + x_offset];
	// os_cfg_reg[6][13] = phosphenes[4 + y_offset][9 + x_offset];
	// os_cfg_reg[6][12] = phosphenes[5 + y_offset][9 + x_offset];
	// os_cfg_reg[6][11] = phosphenes[6 + y_offset][9 + x_offset];

	// //col 12
	// os_cfg_reg[6][10] = phosphenes[0 + y_offset][11 + x_offset];
	// os_cfg_reg[6][9]  = phosphenes[1 + y_offset][11 + x_offset];
	// os_cfg_reg[6][8]  = phosphenes[2 + y_offset][11 + x_offset];
	// os_cfg_reg[6][7]  = phosphenes[3 + y_offset][11 + x_offset];
	// os_cfg_reg[6][6]  = phosphenes[4 + y_offset][11 + x_offset];
	// os_cfg_reg[6][5]  = phosphenes[5 + y_offset][11 + x_offset];
	// os_cfg_reg[6][4]  = phosphenes[6 + y_offset][11 + x_offset];

	// //col 14
	// os_cfg_reg[7][11] = phosphenes[0 + y_offset][13 + x_offset];
	// os_cfg_reg[7][10] = phosphenes[1 + y_offset][13 + x_offset];
	// os_cfg_reg[7][9]  = phosphenes[2 + y_offset][13 + x_offset];
	// os_cfg_reg[7][8]  = phosphenes[3 + y_offset][13 + x_offset];
	// os_cfg_reg[7][7]  = phosphenes[4 + y_offset][13 + x_offset];
	// os_cfg_reg[7][6]  = phosphenes[5 + y_offset][13 + x_offset];
	// os_cfg_reg[7][5]  = phosphenes[6 + y_offset][13 + x_offset];

	// for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+7; phosphene_y_index++)
	// {
	// 	for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+14; phosphene_x_index++)
	// 	{	
	// 		if(phosphenes[phosphene_y_index][phosphene_x_index])
	// 			xil_printf("o");
	// 		else
	// 			xil_printf(".");
	// 		// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
	// 	}
	// 	xil_printf("\r\n");	
	// }
	// imec_phosphene_14x7_output(os_cfg_reg);



	sleep(1);
	// imec_test_configuation_set_pattern(0);
//   stimulation = stimulation + torch.sign(stimulation).detach() - stimulation.detach() # (self-through estimator)
//   stimulation = .5*(stimulation+1)



}

void run_camera_NN(NN_config_t *NN, u8 *srcFrame, u8 *destFrame, u32 width, u32 height, u32 stride, u32 start_pixel_width, u32 start_pixel_height)
{
	XTime execution_start_time, execution_end_time;
	double number_bytes_transferred_external_memory = 0;
	double number_accesses_to_external_memory = 0;
	uint64_t *final_layer_number_of_entries_per_row = malloc(LAYER_13_OUTPUT_ROWS*sizeof(uint64_t));
//	print("Running ...\n\r");

	u32 xcoi, ycoi;
//	Xil_DCacheInvalidateRange((unsigned int) srcFrame, DEMO_MAX_FRAME);
	u8 gray_value;
	u32 lineStart = ((1080/2)-(128/2))*stride;
	u32 xstart = (1920/2) - (128/2);
	// u32 lineStart = 0;
	// u32 xstart = 0;
	// u32 image_index = 0;
	for(ycoi = 0; ycoi < height; ycoi++)
	{
		for(xcoi = xstart*3 ; xcoi < ((xstart+width) * 3); xcoi+=3)
		{
			gray_value =  (srcFrame[xcoi + lineStart] + srcFrame[xcoi + lineStart + 1] + srcFrame[xcoi + lineStart + 2])/3;
			// gray_value = srcFrame[image_index];
			// image_index += 1;
			destFrame[xcoi + lineStart] = gray_value;     //Red
			destFrame[xcoi + lineStart + 1] = gray_value; //Blue
			destFrame[xcoi + lineStart + 2] = gray_value; //Green
		}
		lineStart += stride;
	}
	Xil_DCacheFlush();
//	Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
//	to_Gray_Scale(srcFrame, destFrame, width, height, stride, start_pixel_width, start_pixel_height);

	XTime_GetTime(&execution_start_time);

	// This function is incorrect. TODO:: fix me
	compressed_input_image(srcFrame, NN->layer_0->input_activations, width, height, stride, start_pixel_width, start_pixel_height);
	// NN->layer_0->input_activations = compressed_src_frame;
	Xil_DCacheFlush();

	execute_layer(NN->axi_cdma_instance, NN->layer_0 , NN->layer_1->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 0 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_1 , NN->layer_2->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 1 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_2 , NN->layer_3->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 2 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_3 , NN->layer_4->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 3 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_4 , NN->layer_5->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 4 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_5 , NN->layer_6->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 5 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_6 , NN->layer_7->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);\
//	print("layer 6 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_7 , NN->layer_8->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 7 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_8 , NN->layer_9->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 8 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_9 , NN->layer_10->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 9 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_10, NN->layer_11->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 10 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_11, NN->layer_12->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 11 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_12, NN->layer_13->number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 12 \n\r");
	execute_layer(NN->axi_cdma_instance, NN->layer_13, final_layer_number_of_entries_per_row, &number_bytes_transferred_external_memory, &number_accesses_to_external_memory);
//	print("layer 13 \n\r");


	XTime_GetTime(&execution_end_time);
	print("Summary: \n\r");
	printf("Total run time = %.7fs\n\r", 1.0 * (execution_end_time - execution_start_time) / COUNTS_PER_SECOND);
	printf("Bytes transferred = %.1f \n\r", number_bytes_transferred_external_memory);
	printf("Number of transfer requests = %.1f \n\r", number_accesses_to_external_memory);

	Xil_DCacheFlush();

	s8 value;
	s8 sign;
	float gray_value_float;
	lineStart = ((1080/2) + (128/2))*DEMO_STRIDE;
	xstart = (1920/2) + (128/2);
	// lineStart = (128)*DEMO_STRIDE;
	// xstart = 128;
	u8 word_i = 0;
	uint64_t word;
//	printf("word_i=%d, word=%llX \n\r", word_i, word);
	int shift_index = 56;
	u8 sm;
	uint16_t count = 0;
	xcoi = xstart*3;
	ycoi = 0;
	u8 sm_flag = 1;
	u8 phosphenes [32][32] = {0};
	u8 phosphene_x_index = 0;
	u8 phosphene_y_index = 0;
	while(ycoi != 32)
	{
//		printf("ycoi=%d \n\r", ycoi);
		shift_index = 56;
		word = NN->layer_13->output_image[word_i];
		word_i += 1;
//		printf("word_i=%d, word=%llX \n\r", word_i, word);
		while(1)
		{
			if(sm_flag==1){
				sm = word >> shift_index;
//				printf("sm=%X \n\r", sm);
				sm_flag = 0;
				shift_index -= 8;
				if(shift_index==-8){
					break;
				}
			}
			else{
				if(sm==0){
					value   = 0;
					sm_flag = 1;
//					printf("value=%d \n\r", value);
					gray_value = 0;
					destFrame[xcoi + lineStart] = gray_value;     //Red
					destFrame[xcoi + lineStart + 1] = gray_value; //Blue
					destFrame[xcoi + lineStart + 2] = gray_value; //Green
					xcoi += 3;
					phosphene_x_index += 1;
					if(xcoi == (xstart+32)*3){
						xcoi = xstart*3;
						lineStart += DEMO_STRIDE;
						ycoi += 1;
						phosphene_x_index = 0;
						phosphene_y_index += 1;
//						print("\n");
//						print("------- \n\r");
						break;
					}
				}
				else{
					value = word >> shift_index;
//					printf("value=%d \n\r", value);
					sm_flag = 1;
					shift_index -= 8;
					gray_value_float = tanh(value);
					sign = gray_value_float>=0? 1 : -1;
					gray_value_float = gray_value_float + sign - gray_value_float;
					gray_value_float = 0.5*(gray_value_float+1);
					gray_value = gray_value_float*255;
					if(gray_value==255){
						count += 1;
					}
					destFrame[xcoi + lineStart] = gray_value;     //Red
					destFrame[xcoi + lineStart + 1] = gray_value; //Blue
					destFrame[xcoi + lineStart + 2] = gray_value; //Green
					phosphenes[phosphene_y_index][phosphene_x_index] = (int)gray_value_float;
					xcoi += 3;
					phosphene_x_index += 1;
					if(xcoi == (xstart+32)*3){
						xcoi = xstart*3;
						lineStart += DEMO_STRIDE;
						ycoi += 1;
						phosphene_x_index = 0;
						phosphene_y_index += 1;
//						print("\n");
//						print("------- \n\r");
						break;
					}
					if(shift_index==-8){
						break;
					}
				}
			}
		}
	}
//	printf("count=%d \n\r", count);
//	Xil_DCacheFlushRange((unsigned int) outputFrame, DEMO_MAX_FRAME);

//	destFrame = outputFrame;
	// Xil_DCacheFlushRange((unsigned int) destFrame, DEMO_MAX_FRAME);
	Xil_DCacheFlush();

	//// print phosphenes
	for (size_t phosphene_y_index = 0; phosphene_y_index < 32; phosphene_y_index++)
	{
		for (size_t phosphene_x_index = 0; phosphene_x_index < 32; phosphene_x_index++)
		{	
			if(phosphenes[phosphene_y_index][phosphene_x_index])
				xil_printf("o");
			else
				xil_printf(".");
			// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
		}
		xil_printf("\r\n");	
	}	

	
	// //// map phosphenes to LEDs
	// //// 7x5 LED display
	// //// offsets are used to select the region to crop and show on the LEDs
	// int y_offset = 32/2 - 7/2;
	// int x_offset = 32/2 - 5/2; 
	// unsigned char os_cfg_reg[3][16] = {0};
	// os_cfg_reg[0][15] = phosphenes[0 + y_offset][0 + x_offset];
	// os_cfg_reg[0][14] = phosphenes[1 + y_offset][0 + x_offset];
	// os_cfg_reg[0][13] = phosphenes[2 + y_offset][0 + x_offset];
	// os_cfg_reg[0][12] = phosphenes[3 + y_offset][0 + x_offset];
	// os_cfg_reg[0][11] = phosphenes[4 + y_offset][0 + x_offset];
	// os_cfg_reg[0][10] = phosphenes[5 + y_offset][0 + x_offset];
	// os_cfg_reg[0][9]  = phosphenes[6 + y_offset][0 + x_offset];

	// os_cfg_reg[0][7] = phosphenes[0 + y_offset][1 + x_offset];
	// os_cfg_reg[0][6] = phosphenes[1 + y_offset][1 + x_offset];
	// os_cfg_reg[0][5] = phosphenes[2 + y_offset][1 + x_offset];
	// os_cfg_reg[0][4] = phosphenes[3 + y_offset][1 + x_offset];
	// os_cfg_reg[0][3] = phosphenes[4 + y_offset][1 + x_offset];
	// os_cfg_reg[0][2] = phosphenes[5 + y_offset][1 + x_offset];
	// os_cfg_reg[0][1] = phosphenes[6 + y_offset][1 + x_offset];

	// os_cfg_reg[1][15] = phosphenes[0 + y_offset][2 + x_offset];
	// os_cfg_reg[1][14] = phosphenes[1 + y_offset][2 + x_offset];
	// os_cfg_reg[1][13] = phosphenes[2 + y_offset][2 + x_offset];
	// os_cfg_reg[1][12] = phosphenes[3 + y_offset][2 + x_offset];
	// os_cfg_reg[1][11] = phosphenes[4 + y_offset][2 + x_offset];
	// os_cfg_reg[1][10] = phosphenes[5 + y_offset][2 + x_offset];
	// os_cfg_reg[1][9]  = phosphenes[6 + y_offset][2 + x_offset];

	// os_cfg_reg[1][7] = phosphenes[0 + y_offset][3 + x_offset];
	// os_cfg_reg[1][6] = phosphenes[1 + y_offset][3 + x_offset];
	// os_cfg_reg[1][5] = phosphenes[2 + y_offset][3 + x_offset];
	// os_cfg_reg[1][4] = phosphenes[3 + y_offset][3 + x_offset];
	// os_cfg_reg[1][3] = phosphenes[4 + y_offset][3 + x_offset];
	// os_cfg_reg[1][2] = phosphenes[5 + y_offset][3 + x_offset];
	// os_cfg_reg[1][1] = phosphenes[6 + y_offset][3 + x_offset];

	// os_cfg_reg[2][13] = phosphenes[0 + y_offset][4 + x_offset];
	// os_cfg_reg[2][12] = phosphenes[1 + y_offset][4 + x_offset];
	// os_cfg_reg[2][11] = phosphenes[2 + y_offset][4 + x_offset];
	// os_cfg_reg[2][10] = phosphenes[3 + y_offset][4 + x_offset];
	// os_cfg_reg[2][9]  = phosphenes[4 + y_offset][4 + x_offset];
	// os_cfg_reg[2][8]  = phosphenes[5 + y_offset][4 + x_offset];
	// os_cfg_reg[2][7]  = phosphenes[6 + y_offset][4 + x_offset];
	// for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+7; phosphene_y_index++)
	// {
	// 	for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+5; phosphene_x_index++)
	// 	{	
	// 		if(phosphenes[phosphene_y_index][phosphene_x_index])
	// 			xil_printf("o");
	// 		else
	// 			xil_printf(".");
	// 		// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
	// 	}
	// 	xil_printf("\r\n");	
	// }
	// imec_phosphene_7x5_output(os_cfg_reg);

	//// map phosphenes to LEDs
	//// 14x7 LED display
	//// offsets are used to select the region to crop and show on the LEDs
	int y_offset = 32/2 - 14/2;
	int x_offset = 32/2 - 7/2;
	unsigned char os_cfg_reg[8][16] = {0};
	//col 1
	os_cfg_reg[0][15] = phosphenes[0 + y_offset][0 + x_offset];
	os_cfg_reg[4][15] = phosphenes[1 + y_offset][0 + x_offset];
	os_cfg_reg[0][8]  = phosphenes[2 + y_offset][0 + x_offset];
	os_cfg_reg[4][8]  = phosphenes[3 + y_offset][0 + x_offset];
	os_cfg_reg[0][1]  = phosphenes[4 + y_offset][0 + x_offset];
	os_cfg_reg[4][1]  = phosphenes[5 + y_offset][0 + x_offset];
	os_cfg_reg[1][10] = phosphenes[6 + y_offset][0 + x_offset];
	os_cfg_reg[5][10] = phosphenes[7 + y_offset][0 + x_offset];
	os_cfg_reg[1][3]  = phosphenes[8 + y_offset][0 + x_offset];
	os_cfg_reg[5][3]  = phosphenes[9 + y_offset][0 + x_offset];
	os_cfg_reg[2][10] = phosphenes[10 + y_offset][0 + x_offset];
	os_cfg_reg[6][10] = phosphenes[11 + y_offset][0 + x_offset];
	os_cfg_reg[3][11] = phosphenes[12 + y_offset][0 + x_offset];
	os_cfg_reg[7][11] = phosphenes[13 + y_offset][0 + x_offset];

	//col 2
	os_cfg_reg[0][14] = phosphenes[0 + y_offset] [1 + x_offset];
	os_cfg_reg[4][14] = phosphenes[1 + y_offset] [1 + x_offset];
	os_cfg_reg[0][7]  = phosphenes[2 + y_offset] [1 + x_offset];
	os_cfg_reg[4][7]  = phosphenes[3 + y_offset] [1 + x_offset];
	os_cfg_reg[0][0]  = phosphenes[4 + y_offset] [1 + x_offset];
	os_cfg_reg[4][0]  = phosphenes[5 + y_offset] [1 + x_offset];
	os_cfg_reg[1][9]  = phosphenes[6 + y_offset] [1 + x_offset];
	os_cfg_reg[5][9]  = phosphenes[7 + y_offset] [1 + x_offset];
	os_cfg_reg[1][2]  = phosphenes[8 + y_offset] [1 + x_offset];
	os_cfg_reg[5][2]  = phosphenes[9 + y_offset] [1 + x_offset];
	os_cfg_reg[2][9]  = phosphenes[10 + y_offset][1 + x_offset];
	os_cfg_reg[6][9]  = phosphenes[11 + y_offset][1 + x_offset];
	os_cfg_reg[3][10] = phosphenes[12 + y_offset][1 + x_offset];
	os_cfg_reg[7][10] = phosphenes[13 + y_offset][1 + x_offset];

	//col 3
	os_cfg_reg[0][13] = phosphenes[0 + y_offset] [2 + x_offset];
	os_cfg_reg[4][13] = phosphenes[1 + y_offset] [2 + x_offset];
	os_cfg_reg[0][6]  = phosphenes[2 + y_offset] [2 + x_offset];
	os_cfg_reg[4][6]  = phosphenes[3 + y_offset] [2 + x_offset];
	os_cfg_reg[1][15] = phosphenes[4 + y_offset] [2 + x_offset];
	os_cfg_reg[5][15] = phosphenes[5 + y_offset] [2 + x_offset];
	os_cfg_reg[1][8]  = phosphenes[6 + y_offset] [2 + x_offset];
	os_cfg_reg[5][8]  = phosphenes[7 + y_offset] [2 + x_offset];
	os_cfg_reg[1][1]  = phosphenes[8 + y_offset] [2 + x_offset];
	os_cfg_reg[5][1]  = phosphenes[9 + y_offset] [2 + x_offset];
	os_cfg_reg[2][8]  = phosphenes[10 + y_offset][2 + x_offset];
	os_cfg_reg[6][8]  = phosphenes[11 + y_offset][2 + x_offset];
	os_cfg_reg[3][9]  = phosphenes[12 + y_offset][2 + x_offset];
	os_cfg_reg[7][9]  = phosphenes[13 + y_offset][2 + x_offset];

	//col 4
	os_cfg_reg[0][12] = phosphenes[0 + y_offset] [3 + x_offset];
	os_cfg_reg[4][12] = phosphenes[1 + y_offset] [3 + x_offset];
	os_cfg_reg[0][5]  = phosphenes[2 + y_offset] [3 + x_offset];
	os_cfg_reg[4][5]  = phosphenes[3 + y_offset] [3 + x_offset];
	os_cfg_reg[1][14] = phosphenes[4 + y_offset] [3 + x_offset];
	os_cfg_reg[5][14] = phosphenes[5 + y_offset] [3 + x_offset];
	os_cfg_reg[1][7]  = phosphenes[6 + y_offset] [3 + x_offset];
	os_cfg_reg[5][7]  = phosphenes[7 + y_offset] [3 + x_offset];
	os_cfg_reg[1][0]  = phosphenes[8 + y_offset] [3 + x_offset];
	os_cfg_reg[5][0]  = phosphenes[9 + y_offset] [3 + x_offset];
	os_cfg_reg[2][7]  = phosphenes[10 + y_offset][3 + x_offset];
	os_cfg_reg[6][7]  = phosphenes[11 + y_offset][3 + x_offset];
	os_cfg_reg[3][8]  = phosphenes[12 + y_offset][3 + x_offset];
	os_cfg_reg[7][8]  = phosphenes[13 + y_offset][3 + x_offset];

	//col 5
	os_cfg_reg[0][11] = phosphenes[0 + y_offset] [4 + x_offset];
	os_cfg_reg[4][11] = phosphenes[1 + y_offset] [4 + x_offset];
	os_cfg_reg[0][4]  = phosphenes[2 + y_offset] [4 + x_offset];
	os_cfg_reg[4][4]  = phosphenes[3 + y_offset] [4 + x_offset];
	os_cfg_reg[1][13] = phosphenes[4 + y_offset] [4 + x_offset];
	os_cfg_reg[5][13] = phosphenes[5 + y_offset] [4 + x_offset];
	os_cfg_reg[1][6]  = phosphenes[6 + y_offset] [4 + x_offset];
	os_cfg_reg[5][6]  = phosphenes[7 + y_offset] [4 + x_offset];
	os_cfg_reg[2][13] = phosphenes[8 + y_offset] [4 + x_offset];
	os_cfg_reg[6][13] = phosphenes[9 + y_offset] [4 + x_offset];
	os_cfg_reg[2][6]  = phosphenes[10 + y_offset][4 + x_offset];
	os_cfg_reg[6][6]  = phosphenes[11 + y_offset][4 + x_offset];
	os_cfg_reg[3][7]  = phosphenes[12 + y_offset][4 + x_offset];
	os_cfg_reg[7][7]  = phosphenes[13 + y_offset][4 + x_offset];

	//col 6
	os_cfg_reg[0][10] = phosphenes[0 + y_offset] [5 + x_offset];
	os_cfg_reg[4][10] = phosphenes[1 + y_offset] [5 + x_offset];
	os_cfg_reg[0][3]  = phosphenes[2 + y_offset] [5 + x_offset];
	os_cfg_reg[4][3]  = phosphenes[3 + y_offset] [5 + x_offset];
	os_cfg_reg[1][12] = phosphenes[4 + y_offset] [5 + x_offset];
	os_cfg_reg[5][12] = phosphenes[5 + y_offset] [5 + x_offset];
	os_cfg_reg[1][5]  = phosphenes[6 + y_offset] [5 + x_offset];
	os_cfg_reg[5][5]  = phosphenes[7 + y_offset] [5 + x_offset];
	os_cfg_reg[2][12] = phosphenes[8 + y_offset] [5 + x_offset];
	os_cfg_reg[6][12] = phosphenes[9 + y_offset] [5 + x_offset];
	os_cfg_reg[2][5]  = phosphenes[10 + y_offset][5 + x_offset];
	os_cfg_reg[6][5]  = phosphenes[11 + y_offset][5 + x_offset];
	os_cfg_reg[3][6]  = phosphenes[12 + y_offset][5 + x_offset];
	os_cfg_reg[7][6]  = phosphenes[13 + y_offset][5 + x_offset];

	//col 7
	os_cfg_reg[0][9]  = phosphenes[0 + y_offset] [6 + x_offset];
	os_cfg_reg[4][9]  = phosphenes[1 + y_offset] [6 + x_offset];
	os_cfg_reg[0][2]  = phosphenes[2 + y_offset] [6 + x_offset];
	os_cfg_reg[4][2]  = phosphenes[3 + y_offset] [6 + x_offset];
	os_cfg_reg[1][11] = phosphenes[4 + y_offset] [6 + x_offset];
	os_cfg_reg[5][11] = phosphenes[5 + y_offset] [6 + x_offset];
	os_cfg_reg[1][4]  = phosphenes[6 + y_offset] [6 + x_offset];
	os_cfg_reg[5][4]  = phosphenes[7 + y_offset] [6 + x_offset];
	os_cfg_reg[2][11] = phosphenes[8 + y_offset] [6 + x_offset];
	os_cfg_reg[6][11] = phosphenes[9 + y_offset] [6 + x_offset];
	os_cfg_reg[2][4]  = phosphenes[10 + y_offset][6 + x_offset];
	os_cfg_reg[6][4]  = phosphenes[11 + y_offset][6 + x_offset];
	os_cfg_reg[3][5]  = phosphenes[12 + y_offset][6 + x_offset];
	os_cfg_reg[7][5]  = phosphenes[13 + y_offset][6 + x_offset];

	for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+14; phosphene_y_index++)
	{
		for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+7; phosphene_x_index++)
		{	
			if(phosphenes[phosphene_y_index][phosphene_x_index])
				xil_printf("o");
			else
				xil_printf(".");
			// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
		}
		xil_printf("\r\n");	
	}
	imec_phosphene_14x7_output(os_cfg_reg);

	//// map phosphenes to LEDs
	//// 7x14 LED display
	//// offsets are used to select the region to crop and show on the LEDs
	// int y_offset = 0;
	// int x_offset = 0;
	// unsigned char os_cfg_reg[8][16] = {0};
	// //col 1
	// os_cfg_reg[0][15] = phosphenes[0 + y_offset][0 + x_offset];
	// os_cfg_reg[0][14] = phosphenes[1 + y_offset][0 + x_offset];
	// os_cfg_reg[0][13] = phosphenes[2 + y_offset][0 + x_offset];
	// os_cfg_reg[0][12] = phosphenes[3 + y_offset][0 + x_offset];
	// os_cfg_reg[0][11] = phosphenes[4 + y_offset][0 + x_offset];
	// os_cfg_reg[0][10] = phosphenes[5 + y_offset][0 + x_offset];
	// os_cfg_reg[0][9]  = phosphenes[6 + y_offset][0 + x_offset];

	// //col 3
	// os_cfg_reg[0][8] = phosphenes[0 + y_offset][2 + x_offset];
	// os_cfg_reg[0][7] = phosphenes[1 + y_offset][2 + x_offset];
	// os_cfg_reg[0][6] = phosphenes[2 + y_offset][2 + x_offset];
	// os_cfg_reg[0][5] = phosphenes[3 + y_offset][2 + x_offset];
	// os_cfg_reg[0][4] = phosphenes[4 + y_offset][2 + x_offset];
	// os_cfg_reg[0][3] = phosphenes[5 + y_offset][2 + x_offset];
	// os_cfg_reg[0][2] = phosphenes[6 + y_offset][2 + x_offset];

	// //col 5
	// os_cfg_reg[0][1]  = phosphenes[0 + y_offset][4 + x_offset];
	// os_cfg_reg[0][0]  = phosphenes[1 + y_offset][4 + x_offset];
	// os_cfg_reg[1][15] = phosphenes[2 + y_offset][4 + x_offset];
	// os_cfg_reg[1][14] = phosphenes[3 + y_offset][4 + x_offset];
	// os_cfg_reg[1][13] = phosphenes[4 + y_offset][4 + x_offset];
	// os_cfg_reg[1][12] = phosphenes[5 + y_offset][4 + x_offset];
	// os_cfg_reg[1][11] = phosphenes[6 + y_offset][4 + x_offset];

	// //col 7
	// os_cfg_reg[1][10] = phosphenes[0 + y_offset][7 + x_offset];
	// os_cfg_reg[1][9]  = phosphenes[1 + y_offset][7 + x_offset];
	// os_cfg_reg[1][8]  = phosphenes[2 + y_offset][7 + x_offset];
	// os_cfg_reg[1][7]  = phosphenes[3 + y_offset][7 + x_offset];
	// os_cfg_reg[1][6]  = phosphenes[4 + y_offset][7 + x_offset];
	// os_cfg_reg[1][5]  = phosphenes[5 + y_offset][7 + x_offset];
	// os_cfg_reg[1][4]  = phosphenes[6 + y_offset][7 + x_offset];

	// //col 9
	// os_cfg_reg[1][3]  = phosphenes[0 + y_offset][8 + x_offset];
	// os_cfg_reg[1][2]  = phosphenes[1 + y_offset][8 + x_offset];
	// os_cfg_reg[1][1]  = phosphenes[2 + y_offset][8 + x_offset];
	// os_cfg_reg[1][0]  = phosphenes[3 + y_offset][8 + x_offset];
	// os_cfg_reg[2][13] = phosphenes[4 + y_offset][8 + x_offset];
	// os_cfg_reg[2][12] = phosphenes[5 + y_offset][8 + x_offset];
	// os_cfg_reg[2][11] = phosphenes[6 + y_offset][8 + x_offset];

	// //col 11
	// os_cfg_reg[2][10] = phosphenes[0 + y_offset][10 + x_offset];
	// os_cfg_reg[2][9]  = phosphenes[1 + y_offset][10 + x_offset];
	// os_cfg_reg[2][8]  = phosphenes[2 + y_offset][10 + x_offset];
	// os_cfg_reg[2][7]  = phosphenes[3 + y_offset][10 + x_offset];
	// os_cfg_reg[2][6]  = phosphenes[4 + y_offset][10 + x_offset];
	// os_cfg_reg[2][5]  = phosphenes[5 + y_offset][10 + x_offset];
	// os_cfg_reg[2][4]  = phosphenes[6 + y_offset][10 + x_offset];

	// //col 13
	// os_cfg_reg[3][11] = phosphenes[0 + y_offset][12 + x_offset];
	// os_cfg_reg[3][10] = phosphenes[1 + y_offset][12 + x_offset];
	// os_cfg_reg[3][9]  = phosphenes[2 + y_offset][12 + x_offset];
	// os_cfg_reg[3][8]  = phosphenes[3 + y_offset][12 + x_offset];
	// os_cfg_reg[3][7]  = phosphenes[4 + y_offset][12 + x_offset];
	// os_cfg_reg[3][6]  = phosphenes[5 + y_offset][12 + x_offset];
	// os_cfg_reg[3][5]  = phosphenes[6 + y_offset][12 + x_offset];

	// //col 2
	// os_cfg_reg[4][15] = phosphenes[0 + y_offset][1 + x_offset];
	// os_cfg_reg[4][14] = phosphenes[1 + y_offset][1 + x_offset];
	// os_cfg_reg[4][13] = phosphenes[2 + y_offset][1 + x_offset];
	// os_cfg_reg[4][12] = phosphenes[3 + y_offset][1 + x_offset];
	// os_cfg_reg[4][11] = phosphenes[4 + y_offset][1 + x_offset];
	// os_cfg_reg[4][10] = phosphenes[5 + y_offset][1 + x_offset];
	// os_cfg_reg[4][9]  = phosphenes[6 + y_offset][1 + x_offset];

	// //col 4
	// os_cfg_reg[4][8] = phosphenes[0 + y_offset][3 + x_offset];
	// os_cfg_reg[4][7] = phosphenes[1 + y_offset][3 + x_offset];
	// os_cfg_reg[4][6] = phosphenes[2 + y_offset][3 + x_offset];
	// os_cfg_reg[4][5] = phosphenes[3 + y_offset][3 + x_offset];
	// os_cfg_reg[4][4] = phosphenes[4 + y_offset][3 + x_offset];
	// os_cfg_reg[4][3] = phosphenes[5 + y_offset][3 + x_offset];
	// os_cfg_reg[4][2] = phosphenes[6 + y_offset][3 + x_offset];

	// //col 6
	// os_cfg_reg[4][1]  = phosphenes[0 + y_offset][5 + x_offset];
	// os_cfg_reg[4][0]  = phosphenes[1 + y_offset][5 + x_offset];
	// os_cfg_reg[5][15] = phosphenes[2 + y_offset][5 + x_offset];
	// os_cfg_reg[5][14] = phosphenes[3 + y_offset][5 + x_offset];
	// os_cfg_reg[5][13] = phosphenes[4 + y_offset][5 + x_offset];
	// os_cfg_reg[5][12] = phosphenes[5 + y_offset][5 + x_offset];
	// os_cfg_reg[5][11] = phosphenes[6 + y_offset][5 + x_offset];

	// //col 8
	// os_cfg_reg[5][10] = phosphenes[0 + y_offset][7 + x_offset];
	// os_cfg_reg[5][9]  = phosphenes[1 + y_offset][7 + x_offset];
	// os_cfg_reg[5][8]  = phosphenes[2 + y_offset][7 + x_offset];
	// os_cfg_reg[5][7]  = phosphenes[3 + y_offset][7 + x_offset];
	// os_cfg_reg[5][6]  = phosphenes[4 + y_offset][7 + x_offset];
	// os_cfg_reg[5][5]  = phosphenes[5 + y_offset][7 + x_offset];
	// os_cfg_reg[5][4]  = phosphenes[6 + y_offset][7 + x_offset];

	// //col 10
	// os_cfg_reg[5][3]  = phosphenes[0 + y_offset][9 + x_offset];
	// os_cfg_reg[5][2]  = phosphenes[1 + y_offset][9 + x_offset];
	// os_cfg_reg[5][1]  = phosphenes[2 + y_offset][9 + x_offset];
	// os_cfg_reg[5][0]  = phosphenes[3 + y_offset][9 + x_offset];
	// os_cfg_reg[6][13] = phosphenes[4 + y_offset][9 + x_offset];
	// os_cfg_reg[6][12] = phosphenes[5 + y_offset][9 + x_offset];
	// os_cfg_reg[6][11] = phosphenes[6 + y_offset][9 + x_offset];

	// //col 12
	// os_cfg_reg[6][10] = phosphenes[0 + y_offset][11 + x_offset];
	// os_cfg_reg[6][9]  = phosphenes[1 + y_offset][11 + x_offset];
	// os_cfg_reg[6][8]  = phosphenes[2 + y_offset][11 + x_offset];
	// os_cfg_reg[6][7]  = phosphenes[3 + y_offset][11 + x_offset];
	// os_cfg_reg[6][6]  = phosphenes[4 + y_offset][11 + x_offset];
	// os_cfg_reg[6][5]  = phosphenes[5 + y_offset][11 + x_offset];
	// os_cfg_reg[6][4]  = phosphenes[6 + y_offset][11 + x_offset];

	// //col 14
	// os_cfg_reg[7][11] = phosphenes[0 + y_offset][13 + x_offset];
	// os_cfg_reg[7][10] = phosphenes[1 + y_offset][13 + x_offset];
	// os_cfg_reg[7][9]  = phosphenes[2 + y_offset][13 + x_offset];
	// os_cfg_reg[7][8]  = phosphenes[3 + y_offset][13 + x_offset];
	// os_cfg_reg[7][7]  = phosphenes[4 + y_offset][13 + x_offset];
	// os_cfg_reg[7][6]  = phosphenes[5 + y_offset][13 + x_offset];
	// os_cfg_reg[7][5]  = phosphenes[6 + y_offset][13 + x_offset];

	// for (size_t phosphene_y_index = y_offset; phosphene_y_index < y_offset+7; phosphene_y_index++)
	// {
	// 	for (size_t phosphene_x_index = x_offset; phosphene_x_index < x_offset+14; phosphene_x_index++)
	// 	{	
	// 		if(phosphenes[phosphene_y_index][phosphene_x_index])
	// 			xil_printf("o");
	// 		else
	// 			xil_printf(".");
	// 		// xil_printf("%3d ", phosphenes[phosphene_y_index][phosphene_x_index]? 111:0);
	// 	}
	// 	xil_printf("\r\n");	
	// }
	// imec_phosphene_14x7_output(os_cfg_reg);



	sleep(2);
	// imec_test_configuation_set_pattern(0);
//   stimulation = stimulation + torch.sign(stimulation).detach() - stimulation.detach() # (self-through estimator)
//   stimulation = .5*(stimulation+1)



}
