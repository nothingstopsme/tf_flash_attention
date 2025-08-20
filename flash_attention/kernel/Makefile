.PHONY : all install clean

CUTE_include_path := -I./cutlass-3.5.0/include

cu_files := flash_attention.cu

CUDA_gencode :=	-gencode=arch=compute_53,code=\"sm_53,compute_53\"

ifdef INTERNAL_TEST

$(info CUDA_gencode = $(CUDA_gencode))

test_executable := internal_test.out

all:
	@echo "Generating $(test_executable) for internal tests"
	@nvcc -std=c++17 -o $(test_executable) $(cu_files) internal_test.cu \
				$(CUTE_include_path) -DINTERNAL_TEST=$(INTERNAL_TEST) -O2 $(CUDA_gencode)

clean:
	@rm -f $(test_executable)

else

define load_tf_module
"import sys\n
sys.path.extend('$(TF_PACKAGE_SEARCH_PATHS)'.split(':'))\n
import tensorflow as tf"
endef

TF_compile_flags := $(shell python -c "$$(echo $(load_tf_module)"\nprint(' '.join(tf.sysconfig.get_compile_flags()))")")
TF_link_flags := $(shell python -c "$$(echo $(load_tf_module)"\nprint(' '.join(tf.sysconfig.get_link_flags()))")")

define locate_cuda_include_paths
"import re, glob, os\n
pattern = re.compile(r'-L(.+/tensorflow)')\n
flags = '$(TF_link_flags)'\n
include_paths = []\n
for flag in flags.split(' '):\n
	match = pattern.match(flag)\n
	if match:\n
		for path in glob.iglob(os.path.join(match.group(1), '**/cuda/include'), recursive=True):\n
			include_paths.append('-I'+path)\n
print(' '.join(include_paths))"
endef

CUDA_include_paths := $(shell python -c "$$(echo $(locate_cuda_include_paths))")

$(info TF_compile_flags = $(TF_compile_flags))
$(info TF_link_flags = $(TF_link_flags))
$(info CUDA_include_paths = $(CUDA_include_paths))
$(info CUDA_gencode = $(CUDA_gencode))

MODULE_NAME ?= flash_attention

cc_files := sync_methods.cc flash_attention_forward.cc flash_attention_backward.cc
so_file := $(MODULE_NAME).so

all: $(so_file)

ifdef INSTALL_TO
$(info INSTALL_TO = $(INSTALL_TO))

install:
	@cp -f $(so_file) $(INSTALL_TO)/$(so_file)

endif

clean:
	@rm -f $(cu_files:%.cu=%.cu.o)
	@rm -f $(cc_files:%.cc=%.cc.o)
	@rm -f $(so_file)

# Defining an empty rule to avoid triggering the implicit rule for .cu files
%.cu: ;

%.cu.o: %.cu flash_attention.h type_mapping.h type_util.h cute_ext/*.h
	@echo "Generating $@"
	@nvcc -std=c++17 -c -o $@ $< $(CUTE_include_path) $(CUDA_include_paths) $(TF_compile_flags) \
				-DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 $(CUDA_gencode)

%.cc.o: %.cc *.h
	@echo "Generating $@"
	@g++ -std=c++17 -c -o $@ $< $(CUTE_include_path) $(CUDA_include_paths) $(TF_compile_flags) \
				-fPIC -DGOOGLE_CUDA=1 -O2


$(so_file): $(cu_files:%.cu=%.cu.o) $(cc_files:%.cc=%.cc.o)
	@echo "Generating $@"
	@g++ -std=c++17 -shared -o $@ $^ $(TF_link_flags) -O2

endif
