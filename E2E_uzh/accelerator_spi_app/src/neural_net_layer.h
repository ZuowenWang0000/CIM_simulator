/*
*   Institute of Neuroinformatics - Sensors Group - UZH/ETHz
*   Title:
*   Date:   18.03.2022
*   Author: hasan
*   Description:
*/

#ifndef SRC_NEURAL_NET_LAYER_H_
#define SRC_NEURAL_NET_LAYER_H_

#include <stdint.h>
#include <xtime_l.h>
#include <math.h>
#include <sleep.h>
#include <xstatus.h>
#include "xaxicdma.h"
#include "NVP_v1_config.h"
//#include "intc/intc.h"
//#include "test_neural_net.h"

#include "xstatus.h"
/*
 * TODO: check if there is a better preprocessor check to determine
 *       if this is a microblaze or zynq system.
 */
#ifdef XPAR_INTC_0_DEVICE_ID
 #include "xintc.h"
#else
 #include "xscugic.h"
#endif

#ifdef XPAR_INTC_0_DEVICE_ID
 #define INTC_DEVICE_ID	XPAR_INTC_0_DEVICE_ID
 #define INTC		XIntc
 #define INTC_HANDLER	XIntc_InterruptHandler
#else
 #define INTC_DEVICE_ID	XPAR_SCUGIC_SINGLE_DEVICE_ID
 #define INTC		XScuGic
 #define INTC_HANDLER	XScuGic_InterruptHandler
#endif

//#include "xscugic.h"
//#define INTC_DEVICE_ID	XPAR_SCUGIC_SINGLE_DEVICE_ID
//#define INTC		XScuGic
//#define INTC_HANDLER	XScuGic_InterruptHandler

#define RETURN_ON_FAILURE(x) if ((x) != XST_SUCCESS) return XST_FAILURE;

/*
 * Structure for interrupt id, handler and callback reference
 */
//typedef struct {
//	u8 id;
//	XInterruptHandler handler;
//	void *pvCallbackRef;
//	u8 priority; //not used for microblaze, set to 0
//	u8 trigType; //not used for microblaze, set to 0
//} ivt_t;

XStatus fnInitInterruptController(INTC *psIntc);

typedef struct {
	u8 id;
	XInterruptHandler handler;
	void *pvCallbackRef;
	u8 priority; //not used for microblaze, set to 0
	u8 trigType; //not used for microblaze, set to 0
} ivt_t;
void fnEnableInterrupts(INTC *psIntc, const ivt_t *prgsIvt, unsigned int csIVectors);

#define max(x, y) (((x) > (y)) ? (x) : (y))

typedef struct {
	uint32_t config;
} reg_file_config_t;

typedef struct {
	uint16_t input_cols;
	uint16_t input_rows;
	uint16_t input_ch;

	uint16_t kernel_k;

	uint16_t output_cols;
	uint16_t output_rows;
	uint16_t output_ch;

	volatile uint64_t *output_image;

	uint64_t *number_of_entries_per_row;
	uint64_t total_number_of_input_activations_entries;
	uint64_t *input_activations;

	 uint64_t *number_of_entries_per_output_row;
	 uint64_t total_number_of_output_activations_entries;
	 uint64_t *ground_truth_output_activations;

	 uint64_t number_of_entries_per_weight_array;
	 uint64_t *weight_array_0;
	 uint64_t *weight_array_1;
	 uint64_t *weight_array_2;

	 uint64_t  number_of_entries_bias_array;
	 uint64_t *bias_array;

//	reg_file_config_t reg_file;
	 uint32_t number_of_pre_reg;
	 uint32_t number_of_intra_reg;
	 uint32_t *pre_reg;
	 uint32_t *intra_reg;
	uint16_t input_layer;
	uint16_t strided;
} layer_config_t;

typedef struct {
	XAxiCdma *axi_cdma_instance;
	layer_config_t *layer_0;
	layer_config_t *layer_1;
	layer_config_t *layer_2;
	layer_config_t *layer_3;
	layer_config_t *layer_4;
	layer_config_t *layer_5;
	layer_config_t *layer_6;
	layer_config_t *layer_7;
	layer_config_t *layer_8;
	layer_config_t *layer_9;
	layer_config_t *layer_10;
	layer_config_t *layer_11;
	layer_config_t *layer_12;
	layer_config_t *layer_13;
} NN_config_t;

int32_t init_dma();
int8_t  is_dma_initialized();
int32_t close_dma();

void wait_for_data_bus_cdma_transfer();
void wait_for_weight_bus_cdma_transfer();

uint8_t next_command_interrupt_flag;
uint8_t output_line_stored_interrupt_flag;

void next_command_interrupt_callback(void *InstancePtr);
void output_line_stored_interrupt_callback(void *InstancePtr);

int32_t write_register_file(uint32_t *config);
int32_t write_pre_reg(uint32_t *config, uint32_t number_of_pre_reg);
int32_t write_intra_reg(uint32_t *config, uint32_t number_of_intra_reg, uint64_t offset);
int32_t flip_execution_bit(uint32_t *config, uint64_t offset);
int32_t flip_start_stream_readers_bit(uint32_t *config, uint64_t offset);
int32_t read_output_line_end_address();
int32_t write_weights_line(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory);
int32_t write_activations_line(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory);
int32_t read_activations_line(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory);
int transfer_data_cdma(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory);

int32_t wait_for_next_command_interrupt();
int32_t wait_for_output_line_stored_interrupt();
int32_t execute_layer(XAxiCdma *axi_cdma_instance, layer_config_t *layer, uint64_t *next_layer_number_of_entries_per_row, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory);

int32_t check_output(volatile uint64_t *output, uint64_t *ground_truth, int32_t output_length);
int32_t check_output_range(volatile uint64_t *output, uint64_t *ground_truth, uint32_t start, uint32_t end);




#endif /* SRC_NEURAL_NET_LAYER_H_ */
