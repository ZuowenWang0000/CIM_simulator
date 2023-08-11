/*
*   Institute of Neuroinformatics - Sensors Group - UZH/ETHz
*   Title:
*   Date:   18.03.2022
*   Author: hasan
*   Description:
*/

#ifndef SRC_IMEC_SPI_H_
#define SRC_IMEC_SPI_H_
#include "sleep.h"
#include "xspips.h"
#include "neuraviper_v1.h"

#define SPI_DEVICE_ID		XPAR_XSPIPS_0_DEVICE_ID
#define XSpiPs_SendByte(BaseAddress, Data) \
        XSpiPs_Out32((BaseAddress) + (u32)XSPIPS_TXD_OFFSET, (u32)(Data))

#define XSpiPs_RecvByte(BaseAddress) \
		XSpiPs_In32((u32)((BaseAddress) + (u32)XSPIPS_RXD_OFFSET))

XSpiPs spi;

int spi_master_init(void) {
    // initialization
    XSpiPs_Config *spi_config = XSpiPs_LookupConfig(SPI_DEVICE_ID);
    if (!spi_config) {
        return XST_FAILURE;
    }

    if (XST_SUCCESS != XSpiPs_CfgInitialize(&spi,
        spi_config, spi_config->BaseAddress)) {
        return XST_FAILURE;
    }

    // if (XST_SUCCESS != XSpiPs_SelfTest(&spi)) {
    //     return XST_FAILURE;
    // }

    // master mode, active high SCK and phase 0
    // default: clock is active high, sample on first clock edge
    if (XST_SUCCESS != XSpiPs_SetOptions(&spi, XSPIPS_MASTER_OPTION | XSPIPS_MANUAL_START_OPTION | XSPIPS_FORCE_SSELECT_OPTION)) {
        return XST_FAILURE;
    }

    // PCLK/64, otherwise may have shift-by-one-bit error
    if (XST_SUCCESS != XSpiPs_SetClkPrescaler(&spi, XSPIPS_CLK_PRESCALE_64)) {
        return XST_FAILURE;
    }
    // if (XST_SUCCESS != XSpiPs_SetClkPrescaler(&spi, XSPIPS_CLK_PRESCALE_32)) {
    //     return XST_FAILURE;
    // }

    return XST_SUCCESS;
}


s32 modified_XSpiPs_PolledTransfer(XSpiPs *InstancePtr, u8 *SendBufPtr,
				u8 *RecvBufPtr, u32 ByteCount)
{
	u32 StatusReg;
	u32 ConfigReg;
	u32 TransCount;
	u32 CheckTransfer;
	s32 Status_Polled;
	u8 TempData;

	/*
	 * The RecvBufPtr argument can be NULL.
	 */
	Xil_AssertNonvoid(InstancePtr != NULL);
	Xil_AssertNonvoid(SendBufPtr != NULL);
	Xil_AssertNonvoid(ByteCount > 0U);
	Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

	/*
	 * Check whether there is another transfer in progress. Not thread-safe.
	 */
	if (InstancePtr->IsBusy == TRUE) {
		Status_Polled = (s32)XST_DEVICE_BUSY;
	} else {

		/*
		 * Set the busy flag, which will be cleared when the transfer is
		 * entirely done.
		 */
		InstancePtr->IsBusy = TRUE;

		/*
		 * Set up buffer pointers.
		 */
		InstancePtr->SendBufferPtr = SendBufPtr;
		InstancePtr->RecvBufferPtr = RecvBufPtr;

		InstancePtr->RequestedBytes = ByteCount;
		InstancePtr->RemainingBytes = ByteCount;

		// /*
		//  * If manual chip select mode, initialize the slave select value.
		//  */
	    //  if (XSpiPs_IsManualChipSelect(InstancePtr) == TRUE) {
		// 	ConfigReg = XSpiPs_ReadReg(InstancePtr->Config.BaseAddress,
		// 				 XSPIPS_CR_OFFSET);
		// 	/*
		// 	 * Set the slave select value.
		// 	 */
		// 	ConfigReg &= (u32)(~XSPIPS_CR_SSCTRL_MASK);
		// 	ConfigReg |= InstancePtr->SlaveSelect;
		// 	XSpiPs_WriteReg(InstancePtr->Config.BaseAddress,
		// 			 XSPIPS_CR_OFFSET, ConfigReg);
		// }

		/*
		 * Enable the device.
		 */
		// XSpiPs_Enable(InstancePtr);

		while((InstancePtr->RemainingBytes > (u32)0U) ||
			(InstancePtr->RequestedBytes > (u32)0U)) {
			TransCount = 0U;
			/*
			 * Fill the TXFIFO with as many bytes as it will take (or as
			 * many as we have to send).
			 */
			while ((InstancePtr->RemainingBytes > (u32)0U) &&
				((u32)TransCount < (u32)XSPIPS_FIFO_DEPTH)) {
				XSpiPs_SendByte(InstancePtr->Config.BaseAddress,
						*InstancePtr->SendBufferPtr);
				InstancePtr->SendBufferPtr += 1;
				InstancePtr->RemainingBytes--;
				++TransCount;
			}

			/*
			 * If master mode and manual start mode, issue manual start
			 * command to start the transfer.
			 */
			if ((XSpiPs_IsManualStart(InstancePtr) == TRUE)
				&& (XSpiPs_IsMaster(InstancePtr) == TRUE)) {
				ConfigReg = XSpiPs_ReadReg(
						InstancePtr->Config.BaseAddress,
						 XSPIPS_CR_OFFSET);
				ConfigReg |= XSPIPS_CR_MANSTRT_MASK;
				XSpiPs_WriteReg(InstancePtr->Config.BaseAddress,
						 XSPIPS_CR_OFFSET, ConfigReg);
			}

			/*
			 * Wait for the transfer to finish by polling Tx fifo status.
			 */
	        CheckTransfer = (u32)0U;
	        while (CheckTransfer == 0U){
			StatusReg = XSpiPs_ReadReg(
					        InstancePtr->Config.BaseAddress,
						        XSPIPS_SR_OFFSET);
				if ( (StatusReg & XSPIPS_IXR_MODF_MASK) != 0U) {
					/*
					 * Clear the mode fail bit
					 */
					XSpiPs_WriteReg(
						InstancePtr->Config.BaseAddress,
						XSPIPS_SR_OFFSET,
						XSPIPS_IXR_MODF_MASK);
					Status_Polled = (s32)XST_SEND_ERROR;
					goto END;
				}
		        CheckTransfer = (StatusReg &
							XSPIPS_IXR_TXOW_MASK);
		    }

			/*
			 * A transmit has just completed. Process received data and
			 * check for more data to transmit.
			 * First get the data received as a result of the transmit
			 * that just completed. Receive data based on the
			 * count obtained while filling tx fifo. Always get the
			 * received data, but only fill the receive buffer if it
			 * points to something (the upper layer software may not
			 * care to receive data).
			 */
			while (TransCount != (u32)0U) {
				TempData = (u8)XSpiPs_RecvByte(
					InstancePtr->Config.BaseAddress);
				if (InstancePtr->RecvBufferPtr != NULL) {
					*(InstancePtr->RecvBufferPtr) = TempData;
					InstancePtr->RecvBufferPtr += 1;
				}
				InstancePtr->RequestedBytes--;
				--TransCount;
			}
		}

		// /*
		//  * Clear the slave selects now, before terminating the transfer.
		//  */
		// if (XSpiPs_IsManualChipSelect(InstancePtr) == TRUE) {
		// 	ConfigReg = XSpiPs_ReadReg(InstancePtr->Config.BaseAddress,
		// 				XSPIPS_CR_OFFSET);
		// 	ConfigReg |= XSPIPS_CR_SSCTRL_MASK;
		// 	XSpiPs_WriteReg(InstancePtr->Config.BaseAddress,
		// 			 XSPIPS_CR_OFFSET, ConfigReg);
		// }

		/*
		 * Clear the busy flag.
		 */
		InstancePtr->IsBusy = FALSE;

		/*
		 * Disable the device.
		 */
		// XSpiPs_Disable(InstancePtr);
		Status_Polled = (s32)XST_SUCCESS;
	}

	END:
	return Status_Polled;
}

int spi_master_transfer(int msg_len, const unsigned char *tx_buf, const unsigned char *rx_buf) {
    // Xil_DCacheFlush();
    rx_buf = malloc(msg_len);
    memset(rx_buf, 0, msg_len);
    int status;
    // xil_printf("msg_len=%d \r\n", msg_len);
    // xil_printf("(int)(msg_len/2)=%d \r\n", (int)(msg_len/2));
    // xil_printf("tx_buf[0]=%02X \r\n", tx_buf[0]);
    // xil_printf("rx_buf[0]=%02X \r\n", rx_buf[0]);

    
    /*
     * If master mode and manual start mode, issue manual start
     * command to start the transfer.
     */
    u32 ConfigReg;
    // if ((XSpiPs_IsManualStart(&spi) == TRUE)
    // 	&& (XSpiPs_IsMaster(&spi) == TRUE)) {
    //         xil_printf("inside..\r\n");
    // 	ConfigReg = XSpiPs_ReadReg(
    // 			spi.Config.BaseAddress,
    // 			 XSPIPS_CR_OFFSET);
    // 	ConfigReg |= XSPIPS_CR_MANSTRT_MASK;
    // 	XSpiPs_WriteReg(spi.Config.BaseAddress,
    // 			 XSPIPS_CR_OFFSET, ConfigReg);
    // }
    XSpiPs_Enable(&spi);

    XSpiPs_SetSlaveSelect(&spi, 0);

    for (int i = 0; i < (int)(msg_len/2); i++)
    {
        status = modified_XSpiPs_PolledTransfer(&spi, &tx_buf[i*2], &rx_buf[i*2], 2); 
        Xil_AssertNonvoid(status == XST_SUCCESS);
        // usleep(100);

        // for (int j = i*2; j < (i*2)+2; j++)
        // {
        //     xil_printf("tx_buf[j]=%02X \r\n", tx_buf[j]);
        // }
        // for (int j = i; j < i+2; j++)
        // {
        //     xil_printf("rx_buf[j]=%02X \r\n", rx_buf[j]);
        // }
    }
    if(msg_len%2==1){
        status = modified_XSpiPs_PolledTransfer(&spi, &tx_buf[msg_len-1], &rx_buf[msg_len-1], 1); 
        Xil_AssertNonvoid(status == XST_SUCCESS);
    }

    /*
     * Clear the slave selects now, before terminating the transfer.
     */
    // if (XSpiPs_IsManualChipSelect(&spi) == TRUE) {
    // 	ConfigReg = XSpiPs_ReadReg(spi.Config.BaseAddress,
    // 				XSPIPS_CR_OFFSET);
    // 	ConfigReg |= XSPIPS_CR_SSCTRL_MASK;
    // 	XSpiPs_WriteReg(spi.Config.BaseAddress,
    // 			 XSPIPS_CR_OFFSET, ConfigReg);
    // }
    XSpiPs_SetSlaveSelect(&spi, 1);
    XSpiPs_Disable(&spi);
    
    
    // int status = XSpiPs_PolledTransfer(&spi, tx_buf, rx_buf, msg_len);
    // int status = XSpiPs_Transfer(&spi, tx_buf, rx_buf, msg_len);
    // Xil_AssertNonvoid(status == XST_SUCCESS);
    // Xil_AssertNonvoid(memcmp(tx_buf, rx_buf, msg_len) == 0);

    // for (int i = 0; i < msg_len; i++)
    // {
    //     xil_printf("%02X ", rx_buf[i]);
    // }
    // xil_printf("\r\n");
    // xil_printf("------\r\n");
    
    free(rx_buf);
    return XST_SUCCESS;
}

int write_register(int reg_addr, unsigned char reg_value) {
    unsigned char tx_data[4];

    tx_data[0] = WRITE_CMD;
    tx_data[1] = ADDR_HI(reg_addr);
    tx_data[2] = ADDR_LO(reg_addr);
    tx_data[3] = reg_value;

    int status = spi_master_transfer(4, tx_data, NULL);
    return status;
}

int write_register_generic(int reg_addr, unsigned char *reg_value, int number_of_values) {
    unsigned char tx_data[3+number_of_values];

    tx_data[0] = WRITE_CMD;
    tx_data[1] = ADDR_HI(reg_addr);
    tx_data[2] = ADDR_LO(reg_addr);
    for (int i = 0; i < number_of_values; i++)
    {
        tx_data[3+i] = reg_value[i];
    }
    
    int status = spi_master_transfer(3+number_of_values, tx_data, NULL);
    return status;
}

// 02,00,0F,00,00,7F,7F,14,00,00,00,BE,00,00 /*SU1*/
int configure_stimulation_unit(
    int su_id   , // 0 to 7
    int polarity, // 0: anodic pulse first; 1: cathodic pulse first
    int npulse  , // 0 to 255; 0: infinite number of pulses; otherwise, generate npulse pulses and reset SU_TRIG
    int anode   , // 0 to 127
    int cathode , // 0 to 127
    int tpulse  , // 1 to 255, unit: 100us
    int tdly    , // 1 to 255, unit: 100us
    int ton1    , // 0 to 255, unit: 10us
    int toff    , // 0 to 255, unit: 10us
    int ton2    , // 0 to 255, unit: 10us
    int tdis    , // 0 to 255, unit: 100us
    int tdis_end  // 0 to 255, unit: 100us
) {
    unsigned char reg[11];
    reg[0]  = polarity ?  SUx_CONFIG_POL_MASK : 0x00U;
    reg[1]  = npulse    & SUx_NPULSE_MASK;
    reg[2]  = anode     & SUx_DAC_AN_MASK;
    reg[3]  = cathode   & SUx_DAC_CAT_MASK;
    reg[4]  = tpulse    & SUx_TPULSE_MASK;
    reg[5]  = tdly      & SUx_TDLY_MASK;
    reg[6]  = ton1      & SUx_TON1_MASK;
    reg[7]  = toff      & SUx_TOFF_MASK;
    reg[8]  = ton2      & SUx_TON2_MASK;
    reg[9]  = tdis      & SUx_TDIS_MASK;
    reg[10] = tdis_end  & SUx_TDIS_END_MASK;

    if (write_register_generic(  SUx_CONFIG(su_id), reg, 11)) return XST_FAILURE;
    return XST_SUCCESS;
}

int write_output_stage_shift_register(
    int                  reg_group, // 0 to 7
    const unsigned char *reg_value  // array with 16 elements
) {
    unsigned char tx_data[3+16];
    int reg_addr = SR_OS_CFGx(reg_group);

    tx_data[0] = WRITE_CMD;
    tx_data[1] = ADDR_HI(reg_addr);
    tx_data[2] = ADDR_LO(reg_addr);
    memcpy(&tx_data[3], reg_value, 16);

    // for (int j = 0; j < 16; j++)
    // {
    //     xil_printf("reg_value[j]=%02X \r\n", reg_value[j]);
    // }
    // for (int j = 0; j < 3+16; j++)
    // {
    //     xil_printf("tx_data[j]=%02X \r\n", tx_data[j]);
    // }

    int status = spi_master_transfer(3+16, tx_data, NULL);
    return status;
}

int imec_test_configuation() {
    // configure gen SR register
    
    xil_printf("/*gen SR*/ \r\n");
    unsigned char gen_SR_register_config[2] = {0x05U, 0x11U};
    write_register_generic(SR_GEN_CFG, gen_SR_register_config, 2);

    // enable record-stimulation mode
    xil_printf("/*rec, stim*/ \r\n");
    write_register(OP_MODE, OP_MODE_REC_STIM_MASK);

    // set all triggers to false
    xil_printf("/*no trigger*/ \r\n");
    write_register(SU_TRIG, 0x00U);

    // configure SU SR config register
    xil_printf("/*SU SR config*/ \r\n");
    unsigned char SU_SR_register_config[4] = {0x22U, 0x22U, 0x22U, 0x22U};
    write_register_generic( SR_SU_CFG, SU_SR_register_config, 4);

    // configure SU 1
    xil_printf("/*SU 1*/ \r\n");
    configure_stimulation_unit(
        /* su_id*/   0, /*polarity*/   0x80,
        /*npulse*/0x00, /*   anode*/0x7F, /*cathode*/0x7F,
        /*tpulse*/0x14, /*    tdly*/0x00, /*   ton1*/0x00,
        /*  toff*/0x00, /*    ton2*/0xBE, /*   tdis*/0x00, /*tdis_end*/0x00);
    
    // set all triggers to true
    xil_printf("/*trigger all*/ \r\n");
    write_register(SU_TRIG, 0xFFU);
    return XST_SUCCESS;
}

int imec_phosphene_7x5_output(unsigned char os_cfg_reg[3][16]) {

    for (int i=0; i<3; ++i) {
        // for (int j = 0; j < 16; j++)
        // {
        //     xil_printf("os_cfg_reg[%d][%d]=%02X \r\n", i,j,os_cfg_reg[i][j]);
        // }
        write_output_stage_shift_register(i, os_cfg_reg[i]);
        // sleep(1);
    }
}

int imec_phosphene_14x7_output(unsigned char os_cfg_reg[8][16]) {

    for (int i=0; i<8; ++i) {
        // for (int j = 0; j < 16; j++)
        // {
        //     xil_printf("os_cfg_reg[%d][%d]=%02X \r\n", i,j,os_cfg_reg[i][j]);
        // }
        write_output_stage_shift_register(i, os_cfg_reg[i]);
        // sleep(1);
    }
}

int imec_test_configuation_set_pattern(int pattern) {
    // OS config register values
    if(pattern==0){
        xil_printf("/*circle dot*/ \r\n");
        unsigned char os_cfg_reg[3][16] = {
            {00,01,00,00,00,00,00,01,00,01,01,01,01,01,01,01}, 
            {00,01,00,00,00,00,00,01,00,01,00,00,01,00,00,01}, 
            {00,00,00,00,00,00,00,01,01,01,01,01,01,01,00,00}};
            // configure OS shift registers
            for (int i=0; i<3; ++i) {
                write_output_stage_shift_register(i, os_cfg_reg[i]);
                // sleep(1);
            }
    } 
    else {
        if(pattern==1){
            xil_printf("/*V*/ \r\n");
            unsigned char os_cfg_reg[3][16] = {
                {00,00,01,00,00,00,00,00,00,00,00,01,01,01,01,01},
                {00,00,01,00,00,00,00,00,00,01,00,00,00,00,00,00},
                {00,00,00,00,00,00,00,00,00,01,01,01,01,01,00,00}};
            // configure OS shift registers
            for (int i=0; i<3; ++i) {
                write_output_stage_shift_register(i, os_cfg_reg[i]);
                // sleep(1);
            }
        }
        else{
            xil_printf("/*all high*/ \r\n");
            unsigned char os_cfg_reg[8][16] = {
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01}, 
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01}, 
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01},
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01},
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01},
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01},
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01},
                {01,01,01,01,01,01,01,01,01,01,01,01,01,01,01,01}};
            // configure OS shift registers
            for (int i=0; i<8; ++i) {
                write_output_stage_shift_register(i, os_cfg_reg[i]);
                // sleep(1);
            }
        }
    }
    return XST_SUCCESS;
}

// int main() {
//     init_platform();
//     if (XST_SUCCESS != spi_master_init()) {
// 		return XST_FAILURE;
// 	}

//     // xil_printf("\r\n=====================================================\r\n");
//     // xil_printf("Running simple test ... !\r\n");
//     // if (XST_SUCCESS != spi_test()) {
//     //     xil_printf("SPI test FAILED\r\n");
//     //     return XST_FAILURE;
//     // } else {
//     //     xil_printf("SPI test PASSED\r\n");
//     // }

//     xil_printf("\r\n=====================================================\r\n");
// 	xil_printf("Configuring IMEC's ASIC ... !\r\n");
//     if (XST_SUCCESS != imec_configuation()) {
//         xil_printf("FAILED\r\n");
//         return XST_FAILURE;
//     } else {
//         xil_printf("Finished\r\n");
//     }

//     cleanup_platform();
//     return XST_SUCCESS;
// }





#endif /* SRC_IMEC_SPI_H_ */
