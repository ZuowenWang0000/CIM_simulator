///******************************************************************************
//*
//* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
//*
//* Permission is hereby granted, free of charge, to any person obtaining a copy
//* of this software and associated documentation files (the "Software"), to deal
//* in the Software without restriction, including without limitation the rights
//* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//* copies of the Software, and to permit persons to whom the Software is
//* furnished to do so, subject to the following conditions:
//*
//* The above copyright notice and this permission notice shall be included in
//* all copies or substantial portions of the Software.
//*
//* Use of the Software is limited solely to applications:
//* (a) running on a Xilinx device, or
//* (b) that interact with a Xilinx device through a bus or interconnect.
//*
//* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
//* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//* SOFTWARE.
//*
//* Except as contained in this notice, the name of the Xilinx shall not be used
//* in advertising or otherwise to promote the sale, use or other dealings in
//* this Software without prior written authorization from Xilinx.
//*
//******************************************************************************/
//
///*
// * helloworld.c: simple test application
// *
// * This application configures UART 16550 to baud rate 9600.
// * PS7 UART (Zynq) is not initialized by this application, since
// * bootrom/bsp configures it to baud rate 115200
// *
// * ------------------------------------------------
// * | UART TYPE   BAUD RATE                        |
// * ------------------------------------------------
// *   uartns550   9600
// *   uartlite    Configurable only in HW design
// *   ps7_uart    115200 (configured by bootrom/bsp)
// */
//
//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//#include "platform.h"
//#include "xparameters.h"
//#include "xspips.h"
//#include "xil_printf.h"
//#include "xil_assert.h"
//#include "sleep.h"
//
//#include "neuraviper_v1.h"
//
//#define SPI_DEVICE_ID		XPAR_XSPIPS_0_DEVICE_ID
//
//XSpiPs spi;
//
//int spi_master_init(void) {
//    // initialization
//    XSpiPs_Config *spi_config = XSpiPs_LookupConfig(SPI_DEVICE_ID);
//    if (!spi_config) {
//        return XST_FAILURE;
//    }
//
//    if (XST_SUCCESS != XSpiPs_CfgInitialize(&spi,
//        spi_config, spi_config->BaseAddress)) {
//        return XST_FAILURE;
//    }
//
//    if (XST_SUCCESS != XSpiPs_SelfTest(&spi)) {
//        return XST_FAILURE;
//    }
//
//    // master mode, active high SCK and phase 0
//    // default: clock is active high, sample on first clock edge
//    if (XST_SUCCESS != XSpiPs_SetOptions(&spi, XSPIPS_MASTER_OPTION)) {
//        return XST_FAILURE;
//    }
//
//    // PCLK/64, otherwise may have shift-by-one-bit error
//    if (XST_SUCCESS != XSpiPs_SetClkPrescaler(&spi, XSPIPS_CLK_PRESCALE_64)) {
//        return XST_FAILURE;
//    }
//
//    return XST_SUCCESS;
//}
//
//int spi_master_transfer(int msg_len, const unsigned char *tx_buf, const unsigned char *rx_buf) {
//#ifdef NDEBUG
//    return XSpiPs_PolledTransfer(&spi, tx_buf, rx_buf, msg_len);
//#else
//    rx_buf = malloc(msg_len);
//    memset(rx_buf, 0, msg_len);
//    int status = XSpiPs_PolledTransfer(&spi, tx_buf, rx_buf, msg_len);
//    Xil_AssertNonvoid(status == XST_SUCCESS);
//    Xil_AssertNonvoid(memcmp(tx_buf, rx_buf, msg_len) == 0);
//    free(rx_buf);
//    return XST_SUCCESS;
//#endif // ifdef NDEBUG
//}
//
//int write_register(int reg_addr, unsigned char reg_value) {
//    unsigned char tx_data[4];
//
//    tx_data[0] = WRITE_CMD;
//    tx_data[1] = ADDR_HI(reg_addr);
//    tx_data[2] = ADDR_LO(reg_addr);
//    tx_data[3] = reg_value;
//
//    int status = spi_master_transfer(4, tx_data, NULL);
//    return status;
//}
//
//int configure_stimulation_unit(
//    int su_id   , // 0 to 7
//    int polarity, // 0: anodic pulse first; 1: cathodic pulse first
//    int npulse  , // 0 to 255; 0: infinite number of pulses; otherwise, generate npulse pulses and reset SU_TRIG
//    int anode   , // 0 to 127
//    int cathode , // 0 to 127
//    int tpulse  , // 1 to 255, unit: 100us
//    int tdly    , // 1 to 255, unit: 100us
//    int ton1    , // 0 to 255, unit: 10us
//    int toff    , // 0 to 255, unit: 10us
//    int ton2    , // 0 to 255, unit: 10us
//    int tdis    , // 0 to 255, unit: 100us
//    int tdis_end  // 0 to 255, unit: 100us
//) {
//    if (write_register(  SUx_CONFIG(su_id), polarity ? SUx_CONFIG_POL_MASK : 0x00U)) return XST_FAILURE;
//    if (write_register(  SUx_NPULSE(su_id),   npulse &   SUx_NPULSE_MASK))           return XST_FAILURE;
//    if (write_register(  SUx_DAC_AN(su_id),    anode &   SUx_DAC_AN_MASK))           return XST_FAILURE;
//    if (write_register( SUx_DAC_CAT(su_id),  cathode &  SUx_DAC_CAT_MASK))           return XST_FAILURE;
//    if (write_register(  SUx_TPULSE(su_id),   tpulse &   SUx_TPULSE_MASK))           return XST_FAILURE;
//    if (write_register(    SUx_TDLY(su_id),     tdly &     SUx_TDLY_MASK))           return XST_FAILURE;
//    if (write_register(    SUx_TON1(su_id),     ton1 &     SUx_TON1_MASK))           return XST_FAILURE;
//    if (write_register(    SUx_TOFF(su_id),     toff &     SUx_TOFF_MASK))           return XST_FAILURE;
//    if (write_register(    SUx_TON2(su_id),     ton2 &     SUx_TON2_MASK))           return XST_FAILURE;
//    if (write_register(    SUx_TDIS(su_id),     tdis &     SUx_TDIS_MASK))           return XST_FAILURE;
//    if (write_register(SUx_TDIS_END(su_id), tdis_end & SUx_TDIS_END_MASK))           return XST_FAILURE;
//    return XST_SUCCESS;
//}
//
//int write_output_stage_shift_register(
//    int                  reg_group, // 0 to 7
//    const unsigned char *reg_value  // array with 16 elements
//) {
//    unsigned char tx_data[3+16];
//    int reg_addr = SR_OS_CFGx(reg_group);
//
//    tx_data[0] = WRITE_CMD;
//    tx_data[1] = ADDR_HI(reg_addr);
//    tx_data[2] = ADDR_LO(reg_addr);
//    memcpy(&tx_data[3], reg_value, 16);
//
//    int status = spi_master_transfer(3+16, tx_data, NULL);
//    return status;
//}
//
//int spi_test(void) {
//    if (XST_SUCCESS != spi_master_init()) {
//        return XST_FAILURE;
//    }
//
//     // send and receive message
//     unsigned char *msg = "Hello SPI!\n";
//     unsigned char recv_byte;
//     int correct = 1;
//     for (unsigned char *ptr=msg; *ptr; ++ptr) {
//         if (XST_SUCCESS != XSpiPs_PolledTransfer(&spi, ptr, &recv_byte, 1)) {
//             return XST_FAILURE;
//         }
//         xil_printf("Tx [%c][0x%02X] Rx[%c][0x%02X]\r\n", *ptr, *ptr, recv_byte, recv_byte);
//
//         if (*ptr != recv_byte) {
//             correct = 0;
//         }
//     }
//
//     if (!correct) {
//         return XST_FAILURE;
//     }
//
//    // ASIC protocol
//    xil_printf("Testing ASIC protocal\r\n");
//
//#ifdef NDEBUG
//    xil_printf("No testing is performed\r\n");
//#endif // ifdef NDEBUG
//
//    // enable stimulation mode
//    write_register(OP_MODE, OP_MODE_STIM_MASK);
//
//    // configure SU
//    configure_stimulation_unit(
//        /* su_id*/   0, /*polarity*/   1,
//        /*npulse*/0xF0, /*   anode*/0xF1, /*cathode*/0xF2,
//        /*tpulse*/0xF3, /*    tdly*/0xF4, /*   ton1*/0xF5,
//        /*  toff*/0xF6, /*    ton2*/0xF7, /*   tdis*/0xF8, /*tdis_end*/0xF9);
//    configure_stimulation_unit(
//        /* su_id*/   7, /*polarity*/   0,
//        /*npulse*/0xB0, /*   anode*/0xB1, /*cathode*/0xB2,
//        /*tpulse*/0xB3, /*    tdly*/0xB4, /*   ton1*/0xB5,
//        /*  toff*/0xB6, /*    ton2*/0xB7, /*   tdis*/0xB8, /*tdis_end*/0xB9);
//
//    // OS config register values
//    unsigned char os_cfg_reg[8][16] = {0};
//    for (int i=0; i<8; ++i) {
//        for (int j=0; j<16; ++j) {
//            os_cfg_reg[i][j] = SR_OS_CFGx_VAL(0, i);
//        }
//    }
//
//    // for each frame
//    for (int n=0; n<2; ++n) {
//        // enable the n-th OS group
//        for (int j=0; j<16; ++j) {
//            os_cfg_reg[n][j] |= SR_OS_CFGx_EN_MASK;
//        }
//
//        // configure OS shift registers
//        for (int i=0; i<8; ++i) {
//            write_output_stage_shift_register(i, os_cfg_reg[i]);
//            // sleep(1);
//        }
//
//        // trigger SU
//        write_register(SU_TRIG, SUx_TRIG_MASK(0) | SUx_TRIG_MASK(7));
//
//        // wait till stimulation done
//        // sleep(...);
//
//        // disable the n-th OS group
//        for (int j=0; j<16; ++j) {
//            os_cfg_reg[n][j] &= ~SR_OS_CFGx_EN_MASK;
//        }
//    }
//
//    // disable stimulation mode
//    write_register(OP_MODE, 0);
//
//    xil_printf("Testing ASIC protocal DONE\r\n");
//
//    return XST_SUCCESS;
//}
//
//int main() {
//    init_platform();
//
//    xil_printf("\r\n=====================================================\r\n");
//    xil_printf("Hello, World!\r\n");
//
//    if (XST_SUCCESS != spi_test()) {
//        xil_printf("SPI test FAILED\r\n");
//        return XST_FAILURE;
//    } else {
//        xil_printf("SPI test PASSED\r\n");
//    }
//
//    cleanup_platform();
//    return XST_SUCCESS;
//}
