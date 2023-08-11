################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/timer_ps/timer_ps.c 

OBJS += \
./src/timer_ps/timer_ps.o 

C_DEPS += \
./src/timer_ps/timer_ps.d 


# Each subdirectory must supply rules for building sources it contributes
src/timer_ps/%.o: ../src/timer_ps/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: ARM v7 gcc compiler'
	arm-none-eabi-gcc -Wall -O0 -g3 -c -fmessage-length=0 -MT"$@" -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard -I/media/hasan/HD_4TB/HD_4TB/vitis_2020_1_workspace/accelerator_spi/export/accelerator_spi/sw/accelerator_spi/standalone_domain/bspinclude/include -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


