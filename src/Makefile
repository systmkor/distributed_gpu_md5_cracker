cpu_SRCS := $(wildcard ./cpu/*.c *.c )
BINS := $(wildcard ../bin/*)

cpu_OBJS := $(cpu_SRCS:.c=.o)
cpu_DEPS := $(cpu_OBJS:.o=.d)

cpu_CFLAGS := -MMD -I"../include"

CFLAGS := $(cpu_CFLAGS)

all: cpu

./cpu/md5cracker: $(cpu_OBJS)

../bin/md5cracker: ./cpu/md5cracker
	$(CP) ./cpu/md5cracker ../bin/md5cracker

cpu: ../bin/md5cracker

clean: clean_objs clean_deps clean_bins

clean_objs:
	@rm -rf $(cpu_OBJS)

clean_deps:
	@rm -rf $(cpu_DEPS)

clean_bins:
	@rm -rf $(bins)

.PHONY: clean clean_objs clean_dfiles clean_bins all cpu
-include $(cpu_DEPS)

