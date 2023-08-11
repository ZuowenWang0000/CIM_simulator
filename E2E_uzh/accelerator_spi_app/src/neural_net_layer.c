 /*
 *   Institute of Neuroinformatics - Sensors Group - UZH/ETHz
 *   Title:
 *   Date:   18.03.2022
 *   Author: hasan
 *   Description:
 */

 #include "neural_net_layer.h"



/*
 * This function enables interrupts and connects interrupt service routines declared in
 * an interrupt vector table
 */
void fnEnableInterrupts(INTC *psIntc, const ivt_t *prgsIvt, unsigned int csIVectors)
{
	next_command_interrupt_flag = 1;
	output_line_stored_interrupt_flag = 1;
	unsigned int isIVector;

	Xil_AssertVoid(psIntc != NULL);
	Xil_AssertVoid(psIntc->IsReady == XIL_COMPONENT_IS_READY);

	/* Hook up interrupt service routines from IVT */
	for (isIVector = 0; isIVector < csIVectors; isIVector++)
	{
#ifdef XPAR_INTC_0_DEVICE_ID
		XIntc_Connect(psIntc, prgsIvt[isIVector].id, prgsIvt[isIVector].handler, prgsIvt[isIVector].pvCallbackRef);
		XIntc_Connect(psIntc, NEXT_COMMAND_INTERRUPT_ID, (Xil_InterruptHandler) next_command_interrupt_callback, (void *) psIntc);
		XIntc_Connect(psIntc, OUTPUT_LINE_STORED_INTERRUPT_ID, (Xil_InterruptHandler) output_line_stored_interrupt_callback, (void *) psIntc);

		/* Enable the interrupt vector at the interrupt controller */
		XIntc_Enable(psIntc, prgsIvt[isIVector].id);
		XIntc_Enable(psIntc, NEXT_COMMAND_INTERRUPT_ID);
		XIntc_Enable(psIntc, OUTPUT_LINE_STORED_INTERRUPT_ID);

#else
		XScuGic_SetPriorityTriggerType(psIntc, prgsIvt[isIVector].id,
				prgsIvt[isIVector].priority, prgsIvt[isIVector].trigType);
		if(isIVector==0){
			XScuGic_SetPriorityTriggerType(psIntc, NEXT_COMMAND_INTERRUPT_ID, 0xA0, 0x3);
			XScuGic_SetPriorityTriggerType(psIntc, OUTPUT_LINE_STORED_INTERRUPT_ID, 0xA0, 0x3);
		}


		XScuGic_Connect(psIntc, prgsIvt[isIVector].id,
					 (Xil_ExceptionHandler)prgsIvt[isIVector].handler, prgsIvt[isIVector].pvCallbackRef);
		if(isIVector==0){
			XScuGic_Connect(psIntc, NEXT_COMMAND_INTERRUPT_ID, (Xil_InterruptHandler) next_command_interrupt_callback, (void *) psIntc);
			XScuGic_Connect(psIntc, OUTPUT_LINE_STORED_INTERRUPT_ID, (Xil_InterruptHandler) output_line_stored_interrupt_callback, (void *) psIntc);
		}

		XScuGic_Enable(psIntc, prgsIvt[isIVector].id);
		if(isIVector==0){
			XScuGic_Enable(psIntc, NEXT_COMMAND_INTERRUPT_ID);
			XScuGic_Enable(psIntc, OUTPUT_LINE_STORED_INTERRUPT_ID);
		}
#endif
	}

	Xil_ExceptionInit();
	// Register the interrupt controller handler with the exception table.
	// This is in fact the ISR dispatch routine, which calls our ISRs
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
				(Xil_ExceptionHandler)INTC_HANDLER,
				psIntc);
	Xil_ExceptionEnable();

}

void next_command_interrupt_callback(void *InstancePtr) {
	next_command_interrupt_flag = 1;
}
void output_line_stored_interrupt_callback(void *InstancePtr) {
	output_line_stored_interrupt_flag = 1;
}

int32_t init_dma(XAxiCdma *axi_cdma_0_instance) {
// 	if(cdma_initialized) return XST_FAILURE;
	//------------------------------------------//
	// Initialize CDMA 0
	//------------------------------------------//
	xil_printf("--- Initialize CDMA ---\n\r");
	print("--- ---\n\r");
	XAxiCdma_Config *CfgPtr;
	int cdma_Status = 0;
	u32 cdma_DeviceId = XPAR_AXI_CDMA_0_DEVICE_ID;
	CfgPtr = XAxiCdma_LookupConfig(cdma_DeviceId);
	if (!CfgPtr) {
		return XST_FAILURE;
	}
	cdma_Status = XAxiCdma_CfgInitialize(axi_cdma_0_instance, CfgPtr,CfgPtr->BaseAddress);
	if (cdma_Status != XST_SUCCESS) {
		xil_printf("cdma 0 cfg Initialization failed \n\r");
		return XST_FAILURE;
	}
	//Disable interrupts, we use polling mode
	XAxiCdma_IntrDisable(axi_cdma_0_instance, XAXICDMA_XR_IRQ_ALL_MASK);

	return 0;
}


XStatus fnInitInterruptController(INTC *psIntc)
{
#ifdef XPAR_INTC_0_DEVICE_ID

	// Init driver instance
	RETURN_ON_FAILURE(XIntc_Initialize(psIntc, INTC_DEVICE_ID));

	// Start interrupt controller
	RETURN_ON_FAILURE(XIntc_Start(psIntc, XIN_REAL_MODE));

#else
	XScuGic_Config *IntcConfig;


	/*
	 * Initialize the interrupt controller driver so that it is ready to
	 * use.
	 */
	IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID);
	if (NULL == IntcConfig) {
		return XST_FAILURE;
	}

	RETURN_ON_FAILURE(XScuGic_CfgInitialize(psIntc, IntcConfig,
					IntcConfig->CpuBaseAddress));
#endif


	return XST_SUCCESS;
}

//int32_t close_dma(XAxiCdma *axi_cdma_0_instance) {
//// 	if(!cdma_initialized) return XST_FAILURE;
//	XAxiCdma_Reset(axi_cdma_0_instance);
////	XAxiCdma_Reset(&axi_cdma_1_instance);
//	while(!XAxiCdma_ResetIsDone(axi_cdma_0_instance));
////	while(!XAxiCdma_ResetIsDone(&axi_cdma_1_instance));
//// 	cdma_initialized = 0;
//	return 0;
//}
//
int32_t wait_for_next_command_interrupt() {
//	if(!interrupt_initialized) return XST_FAILURE;
	while(!next_command_interrupt_flag);
	next_command_interrupt_flag = 0;
	return 0;
}
int32_t wait_for_output_line_stored_interrupt() {
//	if(!interrupt_initialized) return XST_FAILURE;
	while(!output_line_stored_interrupt_flag);
	output_line_stored_interrupt_flag = 0;
	return 0;
}

void wait_for_data_bus_cdma_transfer(XAxiCdma *axi_cdma_0_instance){
// 	if(!cdma_initialized) return XST_FAILURE;
	while (XAxiCdma_IsBusy(axi_cdma_0_instance)) {}
}
void wait_for_weight_bus_cdma_transfer(XAxiCdma *axi_cdma_0_instance){
// 	if(!cdma_initialized) return XST_FAILURE;
	while (XAxiCdma_IsBusy(axi_cdma_0_instance)) {}
}


int transfer_data_cdma(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory) {
// 	if(!cdma_initialized) return XST_FAILURE;
	*number_bytes_transferred_external_memory = *number_bytes_transferred_external_memory + length;
	*number_accesses_to_external_memory = *number_accesses_to_external_memory + 1;
//	int cdma_Status;
	XAxiCdma_SimpleTransfer(axi_cdma_instance, (UINTPTR) source_address,
				(UINTPTR) destination_address, length, NULL, NULL);
	while (XAxiCdma_IsBusy(axi_cdma_instance)) {}
	return 0;
}



int32_t write_weights_line(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory){
// 	if(!cdma_initialized) return XST_FAILURE;
//	XAxiCdma weights_cdma = axi_cdma_0_instance;
	if(transfer_data_cdma(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
		return XST_FAILURE;
	}
	return 0;
}

int32_t write_activations_line(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory){
// 	if(!cdma_initialized) return XST_FAILURE;
//	XAxiCdma activations_cdma = axi_cdma_0_instance;
	if(transfer_data_cdma(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
		return XST_FAILURE;
	}
	return 0;
}

int32_t read_activations_line(XAxiCdma *axi_cdma_instance, uint64_t *source_address, uint64_t destination_address, uint64_t length, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory){
// 	if(!cdma_initialized) return XST_FAILURE;
//	XAxiCdma activations_cdma = axi_cdma_0_instance;
	if(transfer_data_cdma(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
		return XST_FAILURE;
	}
	return 0;
}

int32_t write_register_file(uint32_t *config) {
	for(uint32_t i = 0; i < NUMBER_OF_REGISTERS; i++) {
		// write all registers one by one
		((volatile uint32_t*)CONTROL_AXI_BASE_ADDRESS)[i] = config[i];
	}
	return 0;
}

int32_t write_pre_reg(uint32_t *config, uint32_t number_of_pre_reg) {
	for(uint32_t i = 0; i < number_of_pre_reg; i++) {
		// write all registers one by one
		((volatile uint32_t*)CONTROL_AXI_BASE_ADDRESS)[PRE_REGISTER_LIST[i]] = config[i];
	}
	return 0;
}

int32_t write_intra_reg(uint32_t *config, uint32_t number_of_intra_reg, uint64_t offset) {
	for(uint32_t i = 0; i < number_of_intra_reg; i++) {
		// write all registers one by one
		((volatile uint32_t*)CONTROL_AXI_BASE_ADDRESS)[INTRA_REGISTER_LIST[i]] = config[i+offset];
	}
	return 0;
}

int32_t flip_execution_bit(uint32_t *config, uint64_t offset) {
	uint32_t trigger_reg = config[offset];
	trigger_reg = trigger_reg ^ (1);
	((volatile uint32_t*)CONTROL_AXI_BASE_ADDRESS)[NUMBER_OF_REGISTERS-1] = trigger_reg;
	return 0;
}

int32_t flip_start_stream_readers_bit(uint32_t *config, uint64_t offset) {
	uint32_t trigger_reg = config[offset];
	trigger_reg = trigger_reg ^ (1);
	trigger_reg = trigger_reg ^ (1 << 1);
	((volatile uint32_t*)CONTROL_AXI_BASE_ADDRESS)[NUMBER_OF_REGISTERS-1] = trigger_reg;
	return 0;
}

int32_t read_output_line_end_address() {
	// read one to execution bit index (LSB of last register)
	return ((volatile uint32_t*)CONTROL_AXI_BASE_ADDRESS);
}


 int32_t execute_layer(XAxiCdma *axi_cdma_instance, layer_config_t *layer, uint64_t *next_layer_number_of_entries_per_row, double* number_bytes_transferred_external_memory, double* number_accesses_to_external_memory) {
 	uint64_t *source_address;
 	volatile uint64_t destination_address = 0;
 	uint64_t length = 0;
 	uint64_t intra_reg_offset = 0;
 	uint64_t output_activations_offset = 0;
 	int input_activations_offset = 0;
 	int input_line_buffer_offset = 0;
 	int previous_line_index = 0;
 	int next_line_index = 0;
 	int max_entries = 0;
 	int input_row_index = 0;
 	uint64_t AXI_BYTE_ACCESS_BITS = log2(AXI_BUS_BIT_WIDTH/8);
 	uint64_t OUTPUT_LINE_START_ADDRESS = 0;
 	uint64_t output_line_length = 0;
 	uint64_t output_line_end_address = 0;
 //	uint64_t bank_count = (uint64_t)ACTIVATION_BUFFER_BANK_COUNT;

 //	printf("ACTIVATION_BUFFER_BANK_COUNT=%llu \n\r", ACTIVATION_BUFFER_BANK_COUNT);
 //	printf("ACTIVATION_BUFFER_BANK_COUNT=%d \n\r", ACTIVATION_BUFFER_BANK_COUNT);
 //	printf("AXI_BYTE_ACCESS_BITS=%llu \n\r", AXI_BYTE_ACCESS_BITS);
 //	printf("AXI_BUS_BIT_WIDTH=%llu \n\r", AXI_BUS_BIT_WIDTH);
 //	printf("log2(AXI_BUS_BIT_WIDTH/8)=%llu \n\r", log2(AXI_BUS_BIT_WIDTH/8));
 //	printf("log2(AXI_BUS_BIT_WIDTH/8)=%d \n\r", log2(AXI_BUS_BIT_WIDTH/8));

 	//--------------
 	// Transfer pre-reg configuration
 	//--------------
 	// transfer register file configuration
// 	print("before pre reg \r\n");

 	if(write_pre_reg(layer->pre_reg, layer->number_of_pre_reg)) {
 		return XST_FAILURE;
 	}

// 	print("after pre reg \r\n");
 	intra_reg_offset = 0;
 	if(write_intra_reg(layer->intra_reg, layer->number_of_intra_reg, intra_reg_offset)) {
 		return XST_FAILURE;
 	}

// 	print("after intra reg \r\n");

 	//--------------
 	// Transfer weights
 	//--------------
 	source_address = layer->weight_array_0;
 	destination_address = WEIGHT_LINE_BUFFER_0_START_ADDRESS;
 	length = layer->number_of_entries_per_weight_array * sizeof(int64_t);
 	if(write_weights_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 		return XST_FAILURE;
 	}
 	source_address = layer->weight_array_1;
 	destination_address = WEIGHT_LINE_BUFFER_1_START_ADDRESS;
 	if(write_weights_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 		return XST_FAILURE;
 	}
 	source_address = layer->weight_array_2;
 	destination_address = WEIGHT_LINE_BUFFER_2_START_ADDRESS;
 	if(write_weights_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 		return XST_FAILURE;
 	}

// 	print("after weights \r\n");

 	length = layer->number_of_entries_bias_array * sizeof(int64_t);
 	source_address = layer->bias_array;
 	destination_address = BIAS_LINE_BUFFER_START_ADDRESS;
 	if(write_weights_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 		return XST_FAILURE;
 	}

// 	print("after bias \r\n");
//	for(int i=0; i< layer->output_rows; i++){
//		printf("layer->number_of_entries_per_row[%d] = %d \n\r", i, layer->number_of_entries_per_row[i]);
//	}
// 	while (XAxiCdma_IsBusy(axi_cdma_instance)) {}
//	print("before input \r\n");
 	//--------------
 	// Transfer input activations
 	//--------------
 	source_address = layer->input_activations;
 	destination_address = ACTIVATION_LINE_BUFFER_0_START_ADDRESS;
 	if(layer->input_layer==1)
 		length = 26 * sizeof(int64_t);
 	else
 		length = layer->number_of_entries_per_row[input_row_index] * sizeof(int64_t);

 	if(write_activations_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 		return XST_FAILURE;
 	}
 	if(layer->input_layer==1)
 		input_activations_offset += 26;
 	else
 		input_activations_offset += layer->number_of_entries_per_row[input_row_index];

 	input_row_index++;
 	source_address = layer->input_activations + input_activations_offset;
 	destination_address = ACTIVATION_LINE_BUFFER_1_START_ADDRESS;
 	if(layer->input_layer==1)
 		length = 26 * sizeof(int64_t);
 	else
 		length = layer->number_of_entries_per_row[input_row_index] * sizeof(int64_t);

 	if(write_activations_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 		return XST_FAILURE;
 	}
 	if(layer->input_layer==1)
 		input_activations_offset += 26;
 	else
 		input_activations_offset += layer->number_of_entries_per_row[input_row_index];

 	input_row_index++;

// 	print("after print \r\n");

 	//--------------
 	// Trigger execution
 	//--------------
 	if(flip_execution_bit(layer->intra_reg, intra_reg_offset)) {
 		return XST_FAILURE;
 	}
 	if(flip_start_stream_readers_bit(layer->intra_reg, intra_reg_offset)) {
 		return XST_FAILURE;
 	}
 	intra_reg_offset += layer->number_of_intra_reg;

 	for (int output_row_index = 1; output_row_index < layer->output_rows+1; output_row_index++) {
 //	for (int output_row_index = 1; output_row_index < 2; output_row_index++) {

 		if(output_row_index!=layer->output_rows){


 			for(int lines_to_send = 0; lines_to_send<layer->strided+1; lines_to_send++) {
 				if (input_row_index%6 >= 3) {
 					previous_line_index = input_row_index-3;
 					next_line_index = input_row_index+3;
// 					if (next_line_index >= layer->input_rows)
// 						max_entries = layer->number_of_entries_per_row[previous_line_index];
// 					else
// 						max_entries = max(layer->number_of_entries_per_row[previous_line_index], layer->number_of_entries_per_row[next_line_index]);
 					max_entries = 128;


 					input_line_buffer_offset = max_entries + (ACTIVATION_BUFFER_BANK_COUNT - (max_entries%ACTIVATION_BUFFER_BANK_COUNT));
 					input_line_buffer_offset = input_line_buffer_offset << AXI_BYTE_ACCESS_BITS;
 				}
 				else {
 					input_line_buffer_offset = 0;
 					max_entries = 0;
 				}

 				switch (input_row_index%3){ // 3 activation line buffers
 				case 0:
 					destination_address = ACTIVATION_LINE_BUFFER_0_START_ADDRESS + input_line_buffer_offset;
 					break;
 				case 1:
 					destination_address = ACTIVATION_LINE_BUFFER_1_START_ADDRESS + input_line_buffer_offset;
 					break;
 				case 2:
 					destination_address = ACTIVATION_LINE_BUFFER_2_START_ADDRESS + input_line_buffer_offset;
 					break;
 				default:
 					destination_address = ACTIVATION_LINE_BUFFER_0_START_ADDRESS + input_line_buffer_offset;
 					break;
 				}

 				source_address = layer->input_activations + input_activations_offset;

 				if(input_row_index < layer->input_rows) {
 					if(layer->input_layer==1)
 						length = 26 * sizeof(int64_t);
 					else
 						length = layer->number_of_entries_per_row[input_row_index] * sizeof(int64_t);

 					if(write_activations_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 						return XST_FAILURE;
 					}
 					if(layer->input_layer==1)
 						input_activations_offset += 26;
 					else
 						input_activations_offset += layer->number_of_entries_per_row[input_row_index];

 					input_row_index++;
 				}


 			}

 			if(write_intra_reg(layer->intra_reg, layer->number_of_intra_reg, intra_reg_offset)) {
 				return XST_FAILURE;
 			}
 		}

 		//--------------
 		// Wait for next command interrupt
 		//--------------
 		if(wait_for_next_command_interrupt()) {
 			return XST_FAILURE;
 		}

 		//--------------
 		// Wait for output line stored interrupt
 		//--------------
 		if(wait_for_output_line_stored_interrupt()) {
 			return XST_FAILURE;
 		}

 		//--------------
 		// Trigger execution
 		//--------------
 		if(output_row_index!=layer->output_rows){
 			if(flip_execution_bit(layer->intra_reg, intra_reg_offset)) {
 				return XST_FAILURE;
 			}
 			if(flip_start_stream_readers_bit(layer->intra_reg, intra_reg_offset)) {
 				return XST_FAILURE;
 			}
 			intra_reg_offset += layer->number_of_intra_reg;
 		}


 		//--------------
 		// Read output activation line end address
 		//--------------
 		output_line_end_address = Xil_In32(CONTROL_AXI_BASE_ADDRESS);
 		//--------------
 		// Read output activation line
 		//--------------
 //		OUTPUT_LINE_START_ADDRESS = (ACTIVATION_LINE_BUFFER_3_START_ADDRESS - ACTIVATION_LINE_BUFFER_0_START_ADDRESS) >> AXI_BYTE_ACCESS_BITS;
 //		source_address = ACTIVATION_LINE_BUFFER_3_START_ADDRESS;

 		switch ((output_row_index-1)%3) {
 			case 0:
 				OUTPUT_LINE_START_ADDRESS = (ACTIVATION_LINE_BUFFER_3_START_ADDRESS - ACTIVATION_LINE_BUFFER_0_START_ADDRESS) >> AXI_BYTE_ACCESS_BITS;
 				source_address = ACTIVATION_LINE_BUFFER_3_START_ADDRESS;
 				break;
 			case 1:
 				OUTPUT_LINE_START_ADDRESS = (ACTIVATION_LINE_BUFFER_4_START_ADDRESS - ACTIVATION_LINE_BUFFER_0_START_ADDRESS) >> AXI_BYTE_ACCESS_BITS;
 				source_address = ACTIVATION_LINE_BUFFER_4_START_ADDRESS;
 				break;
 			case 2:
 				OUTPUT_LINE_START_ADDRESS = (ACTIVATION_LINE_BUFFER_5_START_ADDRESS - ACTIVATION_LINE_BUFFER_0_START_ADDRESS) >> AXI_BYTE_ACCESS_BITS;
 				source_address = ACTIVATION_LINE_BUFFER_5_START_ADDRESS;
 				break;
 			default:
 				OUTPUT_LINE_START_ADDRESS = (ACTIVATION_LINE_BUFFER_3_START_ADDRESS - ACTIVATION_LINE_BUFFER_0_START_ADDRESS) >> AXI_BYTE_ACCESS_BITS;
 				source_address = ACTIVATION_LINE_BUFFER_3_START_ADDRESS;
 				break;
 		}

 		output_line_length = output_line_end_address-OUTPUT_LINE_START_ADDRESS + 1;
 		destination_address = layer->output_image + output_activations_offset;
 		length = (output_line_length) * sizeof(int64_t);

// 		printf("next_layer_number_of_entries_per_row[%d] = %d \n\r", output_row_index, next_layer_number_of_entries_per_row[output_row_index-1]);
 		next_layer_number_of_entries_per_row[output_row_index-1] = output_line_length;
// 		printf("next_layer_number_of_entries_per_row[%d] = %d \n\r", output_row_index, next_layer_number_of_entries_per_row[output_row_index-1]);
// 		print("----- \r\n");

 //		XTime execution_start_time, execution_end_time;
 //		XTime_GetTime(&execution_start_time);

 		if(read_activations_line(axi_cdma_instance, source_address, destination_address, length, number_bytes_transferred_external_memory, number_accesses_to_external_memory)) {
 			return XST_FAILURE;
 		}


 //		XTime_GetTime(&execution_end_time);
 //		printf("read output line time = %.7fs\n\r", 1.0 * (execution_end_time - execution_start_time) / COUNTS_PER_SECOND);

 		output_activations_offset += output_line_length;



 		if(output_row_index==layer->output_rows) {
 			break;
 		}



 	}

 	return 0;
 }



 int32_t check_output(volatile uint64_t *output, uint64_t *ground_truth, int32_t output_length) {
 	int32_t return_val = 0;
 	// printf("output_length=%d \n\r", output_length);
 	for(uint32_t i = 0; i < output_length; i++) {
 //	for(uint32_t i = 0; i < 10; i++) {
 //		printf("Output check at index=%d \n\r", i);
 //		printf("ground_truth=%llX, output=%llX \n\r", ground_truth[i], output[i]);
 		if(output[i] != ground_truth[i]) {
 			printf("Output check failed at index=%d \n\r", i);
 			printf("ground_truth=%llX, output=%llX \n\r", ground_truth[i], output[i]);
 			printf("last correct ground_truth=%llX, output=%llX \n\r", ground_truth[i-1], output[i-1]);
 			printf("next ground_truth=%llX, output=%llX \n\r", ground_truth[i+1], output[i+1]);
 			return_val = -1;
 			return return_val;
 		}
 	}
 //	printf("prev ground_truth=%llX, output=%llX \n\r", ground_truth[output_length-1], output[output_length-1]);
 //	printf("next ground_truth=%llX, output=%llX \n\r", ground_truth[output_length+1], output[output_length+1]);
 	return return_val;
 }

 int32_t check_output_range(volatile uint64_t *output, uint64_t *ground_truth, uint32_t start, uint32_t end) {
 	int32_t return_val = 0;
 	for(uint32_t i = start; i < end; i++) {
 		printf("Output check at index=%d \n\r", i);
 //		printf("ground_truth=%llu, output=%llu \n\r", ground_truth[i], output[i]);
 		printf("ground_truth=%llX, output=%llX \n\r", ground_truth[i], output[i]);
 //		printf("ground_truth=%llp, output=%llp \n\r", ground_truth[i], output[i]);
 		if(output[i] != ground_truth[i]) {
 			printf("Output check failed at index=%d \n\r", i);
 //			printf("ground_truth=%llX, output=%llX \n\r", ground_truth[i], output[i]);
 //			printf("ground_truth=%llu, output=%llu \n\r", ground_truth[i], output[i]);
 			return_val = -1;
 			return return_val;
 		}
 	}



 //	printf("equal equal %0d \n\r", (97912846 == 2266367759));

 	return return_val;
 }
