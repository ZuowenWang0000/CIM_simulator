################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
LD_SRCS += \
../src/lscript.ld 

C_SRCS += \
../src/main.c \
../src/neural_net_layer.c \
../src/platform.c \
../src/video_demo.c 

OBJS += \
./src/main.o \
./src/neural_net_layer.o \
./src/platform.o \
./src/video_demo.o 

C_DEPS += \
./src/main.d \
./src/neural_net_layer.d \
./src/platform.d \
./src/video_demo.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: ARM v7 gcc compiler'
	arm-none-eabi-gcc -Wall -O0 -g3 -c -fmessage-length=0 -MT"$@" -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard -I/media/hasan/HD_4TB/HD_4TB/vitis_2020_1_workspace/accelerator_spi/export/accelerator_spi/sw/accelerator_spi/standalone_domain/bspinclude/include -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


