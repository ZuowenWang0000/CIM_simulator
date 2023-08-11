#ifndef NEURAVIPER_V1_H
#define NEURAVIPER_V1_H

// ================ SPI commands ================
#define WRITE_CMD 0x02U
#define READ_CMD  0x03U

// ================ register address definitions ================
// 1-byte global control registers
#define OP_MODE                   0x0000U
#define REC_MOD                   0x0001U
#define EL_IMP_MOD                0x0002U
#define STIM_MOD                  0x0003U
#define CAL_MOD                   0x0004U
#define CH_ARRAY_NRST             0x0006U
#define SU_TRIG                   0x0007U

// SUx (stimulation unit #x) register offsets and base
#define SU_CONFIG_OFFSET          0x0000U
#define SU_NPULSE_OFFSET          0x0001U
#define SU_DAC_AN_OFFSET          0x0002U
#define SU_DAC_CAT_OFFSET         0x0003U
#define SU_TPULSE_OFFSET          0x0004U
#define SU_TDLY_OFFSET            0x0005U
#define SU_TON1_OFFSET            0x0006U
#define SU_TOFF_OFFSET            0x0007U
#define SU_TON2_OFFSET            0x0008U
#define SU_TDIS_OFFSET            0x0009U
#define SU_TDIS_END_OFFSET        0x000AU
#define SU_NUM_REG                0x000BU
#define SU_REG_BASE               0x000FU

// 1-byte SUx (stimulation unit #x) registers, x from 0 to 7
#define SUx_REG_ADDR(x, offset)   (SU_REG_BASE+(x)*SU_NUM_REG+offset  )
#define SUx_CONFIG(x)             SUx_REG_ADDR((x), SU_CONFIG_OFFSET  )
#define SUx_NPULSE(x)             SUx_REG_ADDR((x), SU_NPULSE_OFFSET  )
#define SUx_DAC_AN(x)             SUx_REG_ADDR((x), SU_DAC_AN_OFFSET  )
#define SUx_DAC_CAT(x)            SUx_REG_ADDR((x), SU_DAC_CAT_OFFSET )
#define SUx_TPULSE(x)             SUx_REG_ADDR((x), SU_TPULSE_OFFSET  )
#define SUx_TDLY(x)               SUx_REG_ADDR((x), SU_TDLY_OFFSET    )
#define SUx_TON1(x)               SUx_REG_ADDR((x), SU_TON1_OFFSET    )
#define SUx_TOFF(x)               SUx_REG_ADDR((x), SU_TOFF_OFFSET    )
#define SUx_TON2(x)               SUx_REG_ADDR((x), SU_TON2_OFFSET    )
#define SUx_TDIS(x)               SUx_REG_ADDR((x), SU_TDIS_OFFSET    )
#define SUx_TDIS_END(x)           SUx_REG_ADDR((x), SU_TDIS_END_OFFSET)

// 4-byte SU (stimulation unit) shift registers
#define SR_SU_CFG 0x03EDU

// 16-byte OSx (output stage #x) shift registers, x from 0 to 7
#define SR_OS_CFG_BASE            0x03F5U
#define SR_OS_CFGx(x)             (SR_OS_CFG_BASE+(x))

// 32-byte CHx (channel #x) shift registers, x from 0 to 3
#define SR_CH_CFG_BASE            0x043DU
#define SR_CH_CFGx(x)             (SR_CH_CFG_BASE+(x))

// 2-byte general config shift registers
#define SR_GEN_CFG                0x045DU

// ================ register mask definitions ================
// OP_MODE
#define OP_MODE_PDOWN_MASK        0x80U
#define OP_MODE_REC_MASK          0x40U
#define OP_MODE_ELIMP_MASK        0x20U
#define OP_MODE_STIM_MASK         0x10U
#define OP_MODE_REC_STIM_MASK     0x50U
#define OP_MODE_RESET_MASK        0x04U
#define OP_MODE_CAL_MASK          0x02U
#define OP_MODE_TEST_MASK         0x01U

// REC_MOD
#define REC_MOD_CHNRST_MASK       0x40U
#define REC_MOD_DIGNRST_MASK      0x20U
#define REC_MOD_SRDFLT_MASK       0x10U

// EL_IMP_MOD
#define EL_IMP_MOD_FREQ_MASK      0xE0U
#define EL_IMP_MOD_NRST_MASK      0x10U
#define EL_IMP_MOD_AMP_MASK       0x04U

// STIM_MOD
#define STIM_MOD_RST_MASK         0x80U

// CAL_MOD
#define CAL_MOD_OS_MASK           0x80U
#define CAL_MOD_CH_MASK           0x40U
#define CAL_MOD_ADC_MASK          0x20U
#define CAL_MOD_SU_MASK           0x10U
#define CAL_MOD_DAC_MASK          0x08U
#define CAL_MOD_MAN_MASK          0x04U

// CH_ARRAY_NRST
#define CH_ARRAY_NRST_MASK        0xFFU

// SU_TRIG, x from 0 to 7
#define SUx_TRIG_MASK(x)          (0x01U<<(x))

// SUx_REG
#define SUx_CONFIG_POL_MASK       0x80U
#define SUx_NPULSE_MASK           0xFFU
#define SUx_DAC_AN_MASK           0x7FU
#define SUx_DAC_CAT_MASK          0x7FU
#define SUx_TPULSE_MASK           0xFFU
#define SUx_TDLY_MASK             0xFFU
#define SUx_TON1_MASK             0xFFU
#define SUx_TOFF_MASK             0xFFU
#define SUx_TON2_MASK             0xFFU
#define SUx_TDIS_MASK             0xFFU
#define SUx_TDIS_END_MASK         0xFFU

// SR_OS_CFGx
#define SR_OS_CFGx_EN_MASK        0x01U
#define SR_OS_CFGx_SEL_MASK       0x0EU
#define SR_OS_CFGx_BLK_MASK       0x40U
#define SR_OS_CFGx_EN_SHFT        0x00U
#define SR_OS_CFGx_SEL_SHFT       0x01U

// ================ macros ================
#define ADDR_HI(addr)             (((addr) >> 8) & 0xFF)
#define ADDR_LO(addr)             (((addr) >> 0) & 0xFF)
#define SR_OS_CFGx_VAL(en, su)    ((((en)<<SR_OS_CFGx_EN_SHFT )&SR_OS_CFGx_EN_MASK ) | \
                                   (((su)<<SR_OS_CFGx_SEL_SHFT)&SR_OS_CFGx_SEL_MASK))

#endif // NEURAVIPER_V1_H
