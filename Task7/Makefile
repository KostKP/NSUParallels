# Компилятор
CXX = pgc++
# Флаги компиляции
CXXFLAGS = -acc -lcublas -Minfo=all -std=c++11 -Wall -O2 -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib

# Директория для выходных файлов
OUT_DIR = out

# Размеры матриц
MATRIX_SIZES = 128x128 256x256 512x512 1024x1024

# Генерация всех возможных комбинаций
TARGETS = $(foreach size,$(MATRIX_SIZES),$(OUT_DIR)/TASK_$(size))

# Правило по умолчанию
all: $(TARGETS)

# Правило для компиляции каждого варианта
$(OUT_DIR)/TASK_%:
	$(eval FILENAME=$(notdir $@))
	$(eval MATRIX_SIZE=$(word 1,$(subst _, ,$(subst TASK_,,$(FILENAME)))))
	$(eval MATRIX_N=$(firstword $(subst x, ,$(MATRIX_SIZE))))
	$(eval MATRIX_M=$(lastword $(subst x, ,$(MATRIX_SIZE))))
	$(CXX) $(CXXFLAGS) -gpu=cc70 -DNX=$(MATRIX_N) -DNY=$(MATRIX_M) -o $@_C therm_acc2.cpp

# Создание директории для выходных файлов
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Очистка
clean:
	rm -rf $(OUT_DIR)

# Убедимся, что директория out создана перед компиляцией
$(TARGETS): | $(OUT_DIR)
