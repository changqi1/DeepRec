ifneq ($(KERNELRELEASE),)

MODULE_NAME := ddwo
obj-m += $(MODULE_NAME).o
$(MODULE_NAME)-y += driver/module.o
$(MODULE_NAME)-y += $(patsubst $(M)/%.c,%.o,$(wildcard $(M)/src/DataTypes/*.c))
$(MODULE_NAME)-y += $(patsubst $(M)/%.c,%.o,$(wildcard $(M)/src/*.c))
$(MODULE_NAME)-y += $(patsubst $(M)/%.c,%.o,$(foreach dir,$(shell find $(M)/algorithm -type d),$(wildcard $(dir)/*.c)))
$(MODULE_NAME)-y += libm/e_pow.o libm/e_exp.o libm/sincostab.o libm/s_sin.o libm/e_log.o

ccflags-y += -DKERNEL_MODULE -mhard-float -msse -m64 -I$(M)/include -I$(M)/driver -I$(M)/include/DataTypes -I$(M)/src $(foreach dir,$(shell find $(M)/algorithm -type d),-I$(dir))

else

libm:
	@[ -d lib ] || mkdir lib
	@if [ -f /usr/lib64/libm.a ]; \
	then \
	cp /usr/lib64/libm.a lib/; \
	elif [ -f /usr/lib/x86_64-linux-gnu/libm.a ]; \
	then \
	cp /usr/lib/x86_64-linux-gnu/libm.a lib/; \
	elif [ -f /usr/lib32/libm.a ]; \
	then \
	cp /usr/lib32/libm.a lib/; \
	elif [ -f /usr/lib/libm.a ]; \
	then \
	cp /usr/lib/libm.a lib/; \
	else \
	echo "Cannot find libm.a!"; \
	fi
	@cd lib; ar -x libm.a;cd ..;

all:
	[ -d libm ] || mkdir libm
	cp -rf lib/* libm/
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean

endif
