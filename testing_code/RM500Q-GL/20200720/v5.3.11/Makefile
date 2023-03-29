KERNELDIR := /lib/modules/$(shell uname -r)/build
PWD       := $(shell pwd)

obj-m+=./drivers/usb/serial/option.o
obj-m+=./drivers/usb/serial/usb_wwan.o
obj-m+=./drivers/usb/serial/qcserial.o

modules: clean
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

install: modules
	cp $(PWD)/drivers/usb/serial/*.ko /lib/modules/$(shell uname -r)/kernel/drivers/usb/serial/
	depmod

clean:
	rm -rf *~ .tmp_versions modules.order Module.symvers
	find . -type f -name *~ -o -name *.o -o -name *.ko -o -name *.cmd -o -name *.mod.c |  xargs rm -rf
